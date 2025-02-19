#-*- coding:utf-8 -*-
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import conv2d, max_pool2d
import torchvision.transforms as transforms
from dataloader import FastCIFAR10, Mnist
from Mnist_Spike_loader import Mnist_dataloader
from utils import change_conv_sizes
from CONSTANTS import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, NB_TIMESTEPS
# from utils import load_encoded_MNIST


class SpikingPool:
    """ 
    Pooling layer with spiking neurons that can fire only once.
    """
    def __init__(self, input_shape, kernel_size, stride, padding=0):
        batch_size, in_channels, in_height, in_width = input_shape
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride,stride)
        self.padding = padding if isinstance(padding, tuple) else (padding,padding)
        out_height = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        out_width = int(((in_width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        self.output_shape = (in_channels, out_height, out_width)
        # Keep track of active neurons because they can fire once
        # self.active_neurons = np.ones(self.output_shape).astype(bool)


    # def reset(self):
    #     self.active_neurons[:] = True


    def __call__(self, in_spks):
        # padding 
        in_spks = F.pad(in_spks, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        # Max pooling (using torch as it is fast and easier, to be changed)
        out_spks = max_pool2d(in_spks, self.kernel_size, stride=self.stride)
        # Keep spikes of active neurons
        # out_spks = out_spks * self.active_neurons
        # Update active neurons as each pooling neuron can fire only once
        # self.active_neurons[out_spks == 1] = False
        return out_spks




class SpikingConv:
    """ 
    Convolutional layer with IF spiking neurons that can fire only once.
    Implements a Winner-take-all STDP learning rule.
    """
    def __init__(self, input_shape, out_channels, kernel_size, stride, padding=0, 
                nb_winners=1, firing_threshold=1, stdp_max_iter=None, adaptive_lr=False,
                stdp_a_plus=0.004, stdp_a_minus=-0.003, stdp_a_max=0.15, inhibition_radius=0,
                update_lr_cnt=500, weight_init_mean=0.8, weight_init_std=0.05, v_reset=0, device="cpu"
        ):
        in_channels, in_height, in_width = input_shape
        self.out_channels = out_channels
        self.device = torch.device(device)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride,stride)
        self.padding = padding if isinstance(padding, tuple) else (padding,padding)
        self.firing_threshold = firing_threshold
        self.v_reset = v_reset
        self.weights = torch.normal(
            mean=weight_init_mean,
            std=weight_init_std,
            size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        ).to(self.device)

        # Output neurons
        out_height = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        out_width = int(((in_width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        self.pot = torch.zeros((TEST_BATCH_SIZE, out_channels, out_height, out_width)).to(self.device)
        self.active_neurons = torch.ones(self.pot.shape, dtype=torch.bool).to(self.device)
        self.output_shape = self.pot.shape

        # STDP
        self.recorded_spks = torch.zeros((in_channels, in_height+2*self.padding[0], in_width+2*self.padding[1]))
        self.nb_winners = nb_winners
        self.inhibition_radius = inhibition_radius
        self.adaptive_lr = adaptive_lr
        self.a_plus = stdp_a_plus
        self.a_minus = stdp_a_minus
        self.a_max = stdp_a_max
        self.stdp_cnt = 0
        self.update_lr_cnt = update_lr_cnt
        self.stdp_max_iter = stdp_max_iter
        self.plasticity = True
        self.stdp_neurons = torch.ones(self.pot.shape, dtype=torch.bool)


    def get_learning_convergence(self):
        return (self.weights * (1-self.weights)).sum() / torch.prod(torch.tensor(self.weights.shape))


    def reset(self):
        self.pot[:] = self.v_reset
        self.active_neurons[:] = True
        self.stdp_neurons[:] = True
        self.recorded_spks[:] = 0
        
        
    def get_winners(self):
        winners = []
        channels = np.arange(self.pot.shape[1])
        # Copy potentials and keep neurons that can do STDP
        pots_tmp = self.pot.clone() * self.stdp_neurons
        # Find at most nb_winners
        while len(winners) < self.nb_winners:
            # Find new winner
            winner = torch.argmax(pots_tmp) # 1D index
            winner = np.unravel_index(winner, pots_tmp.shape) # 3D index
            # Assert winner potential is higher than firing threshold
            # If not, stop the winner selection 
            if pots_tmp[winner] <= self.firing_threshold:
                break
            # Add winner
            winners.append(winner)
            # Disable winner selection for neurons in neighborhood of other channels
            # pots_tmp[:,channels != winner[1],
            #     max(0,winner[2]-self.inhibition_radius):winner[1]+self.inhibition_radius+1,
            #     max(0,winner[3]-self.inhibition_radius):winner[2]+self.inhibition_radius+1
            # ] = self.v_reset
            # # Disable winner selection for neurons in same channel
            # pots_tmp[winner[1]] = self.v_reset 
        return winners


    def lateral_inhibition(self, spks):
        # Get index of spikes
        batch_size, spks_c,spks_h,spks_w = np.where(spks)
        # Get associated potentials
        spks_pot = np.array([self.pot[spks_c[i],spks_h[i],spks_w[i]] for i in range(len(spks_c))])
        # Sort index by potential in a descending order
        spks_sorted_ind = np.argsort(spks_pot)[::-1]
        # Sequentially inhibit neurons in the neighborhood of other channels
        # Neurons with highest potential inhibit neurons with lowest one, even if both spike
        for ind in spks_sorted_ind:
            # Check that neuron has not been inhibated by another one
            if spks[spks_c[ind],spks_h[ind],spks_w[ind]] == 1:
                # Compute index
                inhib_channels = np.arange(spks.shape[0]) != spks_c[ind]
                # Inhibit neurons
                spks[inhib_channels,spks_h[ind],spks_w[ind]] = 0 
                self.pot[inhib_channels,spks_h[ind],spks_w[ind]] = self.v_reset
                self.active_neurons[inhib_channels,spks_h[ind],spks_w[ind]] = False
        return spks


    def get_conv_of(self, input, output_neuron):
        # Neuron index
        batch, n_c, n_h, n_w = output_neuron
        # Get the list of convolutions on input neurons to update output neurons
        # shape : (in_neuron_values, nb_convs)
        convs = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, stride=self.stride)[0]
        # Get the convolution for the spiking neuron
        conv_ind = (n_h * self.pot.shape[2]) + n_w # 2D to 1D index
        return convs[:, conv_ind]
    
    # F.unfold 과정에서 stride 가 가능한 경우의 수를 구하는 메서드    
    def get_output_elements(self, height, width):
        output_height = (height-self.kernel_size[0])/self.stride[0] + 1
        output_width = (width-self.kernel_size[1])/self.stride[1] + 1
        return int(output_height * output_width)        
        
    def stdp_(self, x_slice, winner_channel, winner_neuron_spk_time):

        if not self.plasticity: return

        delta_weights_zero = (x_slice == 0) * self.a_minus
        delta_plus_weights_nonzero = (x_slice != 0) * (x_slice <= winner_neuron_spk_time) * self.a_plus * torch.exp(-(winner_neuron_spk_time - x_slice))
        delta_minus_weights_nonzero = (x_slice != 0) * (x_slice > winner_neuron_spk_time) * self.a_minus * torch.exp(winner_neuron_spk_time - x_slice)
        dW = delta_weights_zero + delta_plus_weights_nonzero + delta_minus_weights_nonzero
        dW = dW.to(self.device)
        self.weights[winner_channel] += dW
        
    def stdp(self, winner):
        # if not self.stdp_neurons[winner]: exit(1)
        if not self.plasticity: return
        # Count call
        self.stdp_cnt += 1
        # Winner 3D coordinates
        batch, winner_c, winner_h, winner_w = winner
        # Get convolution window used to compute output neuron potential
        conv = self.get_conv_of(self.recorded_spks, winner).flatten()
        # Compute dW
        w = self.weights[winner_c].flatten() * (1 - self.weights[winner_c]).flatten()
        w_plus = conv > 0 # Pre-then-post
        w_minus = conv == 0 # Post-then-pre (we assume that if no spike before, then after)
        dW = (w_plus * w * self.a_plus) + (w_minus * w * self.a_minus)
        self.weights[winner_c] += dW.reshape(self.weights[winner_c].shape)
        # Lateral inhibition between channels (local inter competition)
        channels = np.arange(self.pot.shape[1])
        self.stdp_neurons[0][channels != winner_c,
            max(0,winner_h-self.inhibition_radius):winner_h+self.inhibition_radius+1,
            max(0,winner_w-self.inhibition_radius):winner_w+self.inhibition_radius+1
        ] = False
        # Lateral inhibition in the same channel (gobal intra competition)
        self.stdp_neurons[0][winner_c] = False
        # Adpative learning rate
        if self.adaptive_lr and self.stdp_cnt % self.update_lr_cnt == 0:
            self.a_plus = min(2 * self.a_plus, self.a_max)
            self.a_minus = - 0.75 * self.a_plus
        # Stop STDP after X trains
        if self.stdp_max_iter is not None and self.stdp_cnt > self.stdp_max_iter:
            self.plasticity = False



    def __call__(self, spk_in, train=False):
        # padding 
        # Keep records of spike input for STDP
        if train == True :
            self.recorded_spks += spk_in
            pass
        else :
            spk_in = F.pad(spk_in, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))  # (left, right, top, bottom)
        in_channel = spk_in.shape[0]
        spk_out = torch.zeros(self.pot.shape)
        
        u = conv2d(spk_in, self.weights, stride=self.stride)
        self.pot[self.active_neurons] += u[self.active_neurons]
        output_spikes = self.pot > self.firing_threshold
        
        if torch.any(output_spikes):
            # Generate spikes
            spk_out[output_spikes] = 1
            # Lateral inhibition for neurons in neighborhood in other channels
            # Inhibit and disable neurons with lower potential that fire
            # spk_out = self.lateral_inhibition(spk_out)
            # STDP plasticity
            if train and self.plasticity:
                # Find winners (based on potential)
                winners = self.get_winners()
                # Apply STDP for each neuron winner
                for winner in winners:
                    self.stdp(winner)
            # Reset potentials and disable neurons that fire
            self.pot[spk_out == 1] = self.v_reset
            self.active_neurons[spk_out == 1] = False
        

        return spk_out

class CSNN(nn.Module):
    """ 
    Convolutional Spiking Neural Network Model.
    """
    def __init__(self, input_shape, device):
        super(CSNN, self).__init__()
    
        conv1 = SpikingConv(input_shape,
            out_channels=30, kernel_size=5, stride=1, padding=2,
            nb_winners=1, firing_threshold=5, stdp_max_iter=None,
            adaptive_lr=True, inhibition_radius=2, v_reset=0, device=device,
        )
        
        pool1 = SpikingPool(conv1.output_shape, kernel_size=2, stride=2, padding=0)

        conv2 = SpikingConv(pool1.output_shape,
            out_channels=100, kernel_size=3, stride=1, padding=1,
            nb_winners=1, firing_threshold=1, stdp_max_iter=None,
            adaptive_lr=True, inhibition_radius=1, v_reset=0, device=device,
        )

        pool2 = SpikingPool(conv2.output_shape, kernel_size=2, stride=2, padding=0)

        conv3 = SpikingConv(pool2.output_shape,
            out_channels=200, kernel_size=3, stride=1, padding=1,
            nb_winners=1, firing_threshold=1, stdp_max_iter=None,
            adaptive_lr=True, inhibition_radius=1, v_reset=0, device=device,
        )

        pool3 = SpikingPool(conv3.output_shape, kernel_size=2, stride=2, padding=0)

        self.conv_layers = [conv1, conv2, conv3]
        self.pool_layers = [pool1, pool2, pool3]
        
        self.output_shape = pool3.output_shape
        self.nb_trainable_layers = len(self.conv_layers)
        self.recorded_sum_spks = []
        

    def reset(self):
        for layer in self.conv_layers:
            layer.reset()


    def __call__(self, x, crop_size=None, train_layer=None):
        self.reset()
        sum_spks = 0
        if train_layer == 0 :
            transform = transforms.RandomCrop((crop_size[train_layer], crop_size[train_layer]))
            x = transform(x) # 1 번째 layer 학습에 맞게 tensor.RandomCrop
        if train_layer == 1 :
            transform = transforms.RandomCrop((crop_size[train_layer], crop_size[train_layer]))
            x = transform(x) # 2 번째 layer 학습에 맞게 tensor.RandomCrop
        if train_layer == 2 :
            transform = transforms.RandomCrop((crop_size[train_layer], crop_size[train_layer]))
            x = transform(x) # 3 번째 layer 학습에 맞게 tensor.RandomCrop
        
        # (B,T,C,H,W) -> (T,B,C,H,W)
        x = x.transpose(0,1)
        timesteps = x.shape[0]
        output_spikes = torch.zeros((timesteps,) + self.output_shape)
        for t in range(timesteps) : 
            # sum_spks += spk_in.sum()
            spk = self.conv_layers[0](x[t], train=(train_layer==0))
            # sum_spks += spk.sum()
            if train_layer == 0 :
                continue
            spk_in = self.pool_layers[0](spk)
            # sum_spks += spk_in.sum()
            spk = self.conv_layers[1](spk_in, train=(train_layer==1))
            # sum_spks += spk.sum()
            if train_layer == 1 :
                continue
            spk_in = self.pool_layers[1](spk)
            # sum_spks += spk_in.sum()
            spk = self.conv_layers[2](spk_in, train=(train_layer==2))
            # sum_spks += spk.sum()
            if train_layer == 2 :
                continue
            spk_out = self.pool_layers[2](spk)
            output_spikes[t] = spk_out
            # sum_spks += spk_out.sum()
            # if train_layer is None:
            #     self.recorded_sum_spks.append(sum_spks)

        return output_spikes


def main(
    seed=1,
    epochs=[1, 1, 1], # Number of epochs per layer
    crop_size = [5, 6, 12],
    convergence_rate=0.01, # Stop training when learning convergence reaches this rate
):
    # use cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:0')
    print('Current cuda device is', device)

    # fixed seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, test_loader, train_loader_for_test, image_shape, train_dataset_len, test_dataset_len, y_train, y_test = Mnist_dataloader(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, NB_TIMESTEPS)


    csnn = CSNN(image_shape, device).to(device)
    temp_csnn = CSNN(image_shape, device).to(device)


    ### TRAINING ###
    print("\n### TRAINING ###")

    for layer in range(csnn.nb_trainable_layers):
        print("Layer {}...".format(layer+1))
        if layer == 0 :
            # 0번 layer 학습용 size로 변경
            change_conv_sizes(csnn, layer, image_shape[0], crop_size[layer], mode="train", device=device) # csnn, 0, 1, 5
        if layer == 1 :
            # 0번 layer와 1번 layer size 변경
            change_conv_sizes(csnn, layer-1, image_shape[0], crop_size[layer], mode="pass", device=device) # csnn, 0, 1, 6
            change_conv_sizes(csnn, layer, csnn.conv_layers[layer-1].out_channels, crop_size[layer]/2, mode="train", device=device) # csnn, 1, 30, 3
        if layer == 2 :
            change_conv_sizes(csnn, layer-2, image_shape[0], crop_size[layer], mode="pass", device=device)
            change_conv_sizes(csnn, layer-1, csnn.conv_layers[layer-2].out_channels, crop_size[layer]/2, mode="pass", device=device)
            change_conv_sizes(csnn, layer, csnn.conv_layers[layer-1].out_channels, crop_size[layer]/4, mode="train", device=device)
        for epoch in range(epochs[layer]):
            print("\t epoch {}".format(epoch + 1))
            for x, _ in tqdm(train_loader):
                x = x.to(device)
                csnn(x, crop_size, train_layer=layer)
                # if csnn.conv_layers[layer].get_learning_convergence() < convergence_rate:
                #     break

    for layer in range(csnn.nb_trainable_layers):
        temp_csnn.conv_layers[layer].weights = csnn.conv_layers[layer].weights.clone().to(device)
    csnn = temp_csnn
    
    ### TESTING ###
    print("\n### TESTING ###")
    
    output_train_max = torch.zeros(train_dataset_len, int(np.prod(csnn.output_shape))).to(device)
    output_train_sum = torch.zeros(train_dataset_len, int(np.prod(csnn.output_shape))).to(device)
    output_train_mean = torch.zeros(train_dataset_len, int(np.prod(csnn.output_shape))).to(device)
    
    for batch_idx, (x, _) in enumerate((tqdm(train_loader_for_test))):
        x = x.to(device)
        # spk.shape : timestep, # of layer3's filter, layer3's filter height, layer3's filter width -> 15, 200, 3, 3
        spk = csnn(x)
        output_train_max[batch_idx] = spk.max(0)[0].flatten()
        output_train_sum[batch_idx] = spk.sum(0).flatten()
        output_train_mean[batch_idx] = spk.mean(0).flatten()
    
    output_test_max = torch.zeros(test_dataset_len, int(np.prod(csnn.output_shape))).to(device)
    output_test_sum = torch.zeros(test_dataset_len, int(np.prod(csnn.output_shape))).to(device)
    output_test_mean = torch.zeros(test_dataset_len, int(np.prod(csnn.output_shape))).to(device)
    
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.to(device)
        spk = csnn(x)
        output_test_max[batch_idx] = spk.max(0)[0].flatten()
        output_test_sum[batch_idx] = spk.sum(0).flatten()
        output_test_mean[batch_idx] = spk.mean(0).flatten()

    # SVM 
    clf = LinearSVC(max_iter=3000, random_state=seed)
    # Move tensors to CPU and convert to NumPy
    output_train_max_np = output_train_max.cpu().numpy()
    output_test_max_np = output_test_max.cpu().numpy()
    
    output_train_sum_np = output_train_sum.cpu().numpy()
    output_test_sum_np = output_test_sum.cpu().numpy()
    
    output_train_mean_np = output_train_mean.cpu().numpy()
    output_test_mean_np = output_test_mean.cpu().numpy()

    y_train_np = y_train
    y_test_np = y_test
    
    # Fit and evaluate SVM
    print("\n### Fit and Calculate Accuracy... ###")
    
    start_time = time.time()
    clf.fit(output_train_max_np, y_train_np)
    y_pred = clf.predict(output_test_max_np)
    acc = accuracy_score(y_test_np, y_pred)
    print("Accuracy(max) : {}".format(acc))
    end_time = time.time()
    print("Time taken for max : {:.4f} seconds".format(end_time - start_time))

    start_time = time.time()
    clf.fit(output_train_sum_np, y_train_np)
    y_pred = clf.predict(output_test_sum_np)
    acc = accuracy_score(y_test_np, y_pred)
    print("Accuracy(sum) : {}".format(acc))
    end_time = time.time()
    print("Time taken for sum : {:.4f} seconds".format(end_time - start_time))

    start_time = time.time()
    clf.fit(output_train_mean_np, y_train_np)
    y_pred = clf.predict(output_test_mean_np)
    acc = accuracy_score(y_test_np, y_pred)
    print("Accuracy(mean) : {}".format(acc))
    end_time = time.time()
    print("Time taken for mean : {:.4f} seconds".format(end_time - start_time))


if __name__ == "__main__":
    main()