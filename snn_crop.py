#-*- coding:utf-8 -*-

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
from utils import Latency_coding, change_conv_sizes, get_sorted_nonzero_indices, get_nonzero_softmax
from CONSTANTS import BATCH_SIZE
# from utils import load_encoded_MNIST


class SpikingPool:
    """ 
    Pooling layer with spiking neurons that can fire only once.
    """
    def __init__(self, input_shape, kernel_size, stride, padding=0):
        in_channels, in_height, in_width = input_shape
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
        in_spks = np.pad(in_spks, ((0,), (self.padding[0],), (self.padding[1],)), mode='constant')
        in_spks = torch.Tensor(in_spks).unsqueeze(0)
        # Max pooling (using torch as it is fast and easier, to be changed)
        out_spks = max_pool2d(in_spks, self.kernel_size, stride=self.stride)[0]
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
        out_height = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1) # 변경대상
        out_width = int(((in_width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1) # 변경대상
        self.pot = torch.zeros((out_channels, out_height, out_width)).to(self.device) # 변경대상
        self.active_neurons = torch.ones(self.pot.shape, dtype=torch.bool).to(self.device) # 변경대상
        self.output_shape = self.pot.shape # 변경대상

        # STDP
        self.recorded_spks = torch.zeros((in_channels, in_height+2*self.padding[0], in_width+2*self.padding[1])) # 변경대상
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
        self.stdp_neurons = torch.ones(self.pot.shape, dtype=torch.bool) # 변경대상


    def get_learning_convergence(self):
        return (self.weights * (1-self.weights)).sum() / torch.prod(torch.tensor(self.weights.shape))


    def reset(self):
        self.pot[:] = self.v_reset
        self.active_neurons[:] = True
        self.stdp_neurons[:] = True
        self.recorded_spks[:] = 0
    
    # F.unfold 과정에서 stride 가 가능한 경우의 수를 구하는 메서드    
    def get_output_elements(self, height, width):
        output_height = (height-self.kernel_size[0])/self.stride[0] + 1
        output_width = (width-self.kernel_size[1])/self.stride[1] + 1
        return int(output_height * output_width)        
        
    def stdp(self, x_slice, winner_channel, winner_neuron_spk_time):

        if not self.plasticity: return

        delta_weights_zero = (x_slice == 0) * self.a_minus
        delta_plus_weights_nonzero = (x_slice != 0) * (x_slice <= winner_neuron_spk_time) * self.a_plus * torch.exp(-(winner_neuron_spk_time - x_slice))
        delta_minus_weights_nonzero = (x_slice != 0) * (x_slice > winner_neuron_spk_time) * self.a_minus * torch.exp(winner_neuron_spk_time - x_slice)
        dW = delta_weights_zero + delta_plus_weights_nonzero + delta_minus_weights_nonzero
        dW = dW.to(self.device)
        self.weights[winner_channel] += dW
        
        # Adpative learning rate
        # if self.adaptive_lr and self.stdp_cnt % self.update_lr_cnt == 0:
        #     self.a_plus = min(2 * self.a_plus, self.a_max)
        
        #     self.a_minus = - 0.75 * self.a_plus



    def __call__(self, spk_in, train=False):
        # padding 
        # Keep records of spike input for STDP
        if train == True :
            # self.recorded_spks += spk_in
            pass
        else :
            spk_in = F.pad(spk_in, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))  # (left, right, top, bottom)
            # spk_in = spk_in.to(self.device)
        in_channel = spk_in.shape[0]
        spk_out = torch.zeros(self.pot.shape)
        spk_out_time = torch.zeros(self.pot.shape)
        reshaped_pot = self.pot.reshape(self.out_channels, self.output_shape[-1]*self.output_shape[-2])
        
        x = spk_in.unsqueeze(0).to(torch.float32)
        # stride 과정 생략 가능한 코드 if 입력이 1,1,10,10 이고 filter 가 5x5 라면 결과 shape = (1,25,36)
        # x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        # ex) x -> x_trans : x_trans.shape : (input channel, stride 가능한 경우의 수, kernal size 만큼의 image pixel 개수)
        x_trans = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride).reshape(in_channel, self.get_output_elements(x.shape[-2],x.shape[-1]), self.kernel_size[0]*self.kernel_size[1])
        sorted_non_zero_indices = get_sorted_nonzero_indices(x_trans)
        
        for i in range (len(sorted_non_zero_indices)):
            self.pot.reshape(self.out_channels, self.output_shape[-1]*self.output_shape[-2])[:,sorted_non_zero_indices[i][1]] += self.weights.reshape(self.out_channels, 
                                                                                                                                                      in_channel,
                                                                                                                                                      self.kernel_size[0]*self.kernel_size[1])[:, sorted_non_zero_indices[i][0],sorted_non_zero_indices[i][2]]
        # self.active_neurons.reshape(self.out_channels, self.output_shape[-1]*self.output_shape[-2])[:,sorted_non_zero_indices[i][1]] 활성화 뉴런 찾는 boolean  tensor
        
            output_spikes = self.pot > self.firing_threshold
            if torch.any(output_spikes):
                # active 가 false 여서 potential 이 0인 뉴런들은 softmax 연산에서 제외되도록 수정 (완)
                self.pot.reshape(self.out_channels, self.output_shape[-1]*self.output_shape[-2])[:,sorted_non_zero_indices[i][1]] = get_nonzero_softmax(self.pot.reshape(self.out_channels, self.output_shape[-1]*self.output_shape[-2])[:,sorted_non_zero_indices[i][1]], dim=0)
                # pot 에서 winner 뉴런의 위치 [?,?,?] 를 알 수 있도록. 근데 어디에 쓸까
                winner_neuron_channel = torch.argmax(self.pot.reshape(self.out_channels, self.output_shape[-1]*self.output_shape[-2])[:,sorted_non_zero_indices[i][1]], dim=0)
                winner_neuron_position_x = sorted_non_zero_indices[i][1] // self.output_shape[-1]
                winner_neuron_position_y = sorted_non_zero_indices[i][1] % self.output_shape[-1]
                
                self.active_neurons[winner_neuron_channel, winner_neuron_position_x, winner_neuron_position_y] = False
                winner_neuron_spk_time = x_trans[sorted_non_zero_indices[i].unbind()]
                spk_out_time[winner_neuron_channel, winner_neuron_position_x, winner_neuron_position_y] = winner_neuron_spk_time
                
                if train and self.plasticity:
                    self.stdp(x_trans[0][sorted_non_zero_indices[i][1]].reshape(self.kernel_size[0],self.kernel_size[1]), winner_neuron_channel, winner_neuron_spk_time)
                
                self.pot[winner_neuron_channel, winner_neuron_position_x, winner_neuron_position_y] = self.v_reset

        return spk_out_time

class CSNN(nn.Module):
    """ 
    Convolutional Spiking Neural Network Model.
    """
    def __init__(self, input_shape, device):
        super(CSNN, self).__init__()
    
        conv1 = SpikingConv(input_shape,
            out_channels=30, kernel_size=5, stride=1, padding=2,
            nb_winners=1, firing_threshold=2.4, stdp_max_iter=None,
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
        

        spk_in = x.type(torch.float64)
        # sum_spks += spk_in.sum()
        spk = self.conv_layers[0](spk_in, train=(train_layer==0))
        # sum_spks += spk.sum()
        if train_layer == 0 :
            return
        spk_in = self.pool_layers[0](spk)
        # sum_spks += spk_in.sum()
        spk = self.conv_layers[1](spk_in, train=(train_layer==1))
        # sum_spks += spk.sum()
        if train_layer == 1 :
            return
        spk_in = self.pool_layers[1](spk)
        # sum_spks += spk_in.sum()
        spk = self.conv_layers[2](spk_in, train=(train_layer==2))
        # sum_spks += spk.sum()
        if train_layer == 2 :
            return
        spk_out = self.pool_layers[2](spk)
        # sum_spks += spk_out.sum()
        # if train_layer is None:
        #     self.recorded_sum_spks.append(sum_spks)

        return spk_out


def main(
    seed=1,
    epochs=[1, 1, 1], # Number of epochs per layer
    crop_size = [5, 6, 12],
    convergence_rate=0.01, # Stop training when learning convergence reaches this rate
    classifier_epochs=20,
    classifier_lr=0.01
):
    dataset = "Mnist"
    # use cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:0')
    print('Current cuda device is', device)

    # fixed seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load datasets
    trainset = Mnist('./data', train=True, download=True)
    y_train = trainset.targets.to(device)
    testset = Mnist('./data', train=False)
    y_test = testset.targets.to(device)

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # Load encoded datasets
    num_train_samples = len(trainset)
    num_test_samples = len(testset)
    if dataset == "Mnist" :
        image_shape = torch.Tensor(trainset[0][0]).unsqueeze(0).shape # 1x28x28
    elif dataset == "FastCifar10" :
        image_shape = trainset[0][0].shape # ex) 3x32x32
    
    encoded_trainset = Latency_coding(num_train_samples, image_shape, trainLoader).to(device)
    encoded_testset = Latency_coding(num_test_samples, image_shape, testLoader).to(device)

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
            for x in tqdm(encoded_trainset):
                x = x.to(device)
                csnn(x, crop_size, train_layer=layer)
                if csnn.conv_layers[layer].get_learning_convergence() < convergence_rate:
                    break

    for layer in range(csnn.nb_trainable_layers):
        temp_csnn.conv_layers[layer].weights = csnn.conv_layers[layer].weights.clone().to(device)
    csnn = temp_csnn
    
    ### TESTING ###
    print("\n### TESTING ###")
    
    output_train_max = torch.zeros((len(encoded_trainset), int(np.prod(csnn.output_shape)))).to(device)
    for i,x in enumerate(tqdm(encoded_trainset)):
        x = x.to(device)
        spk = csnn(x)
        output_train_max[i] = spk.flatten()
    
    output_test_max = torch.zeros((len(encoded_testset), int(np.prod(csnn.output_shape)))).to(device)
    for i,x in enumerate(tqdm(encoded_testset)):
        x = x.to(device)
        spk = csnn(x)
        output_test_max[i] = spk.flatten()

    # SVM 
    clf = LinearSVC(max_iter=3000, random_state=seed)
    # Move tensors to CPU and convert to NumPy
    output_train_max_np = output_train_max.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    output_test_max_np = output_test_max.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # Fit and evaluate SVM
    clf.fit(output_train_max_np, y_train_np)
    y_pred = clf.predict(output_test_max_np)
    acc = accuracy_score(y_test_np, y_pred)
    print("Accuracy with method 1 (max) : {}".format(acc))

if __name__ == "__main__":
    main()