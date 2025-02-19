import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mnist import MNIST

class MNISTSpikeDataset(Dataset):
    def __init__(self, data, labels, nb_timesteps):
        self.data = data
        self.labels = labels
        self.nb_timesteps = nb_timesteps

    def spike_encoding(self, img):
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppress divide by zero warning
            I, lat = np.argsort(1 / img.flatten()), np.sort(1 / img.flatten())
        I = np.delete(I, np.where(lat == np.inf))
        II = np.unravel_index(I, img.shape)
        t_step = np.ceil(np.arange(I.size) / (I.size / (self.nb_timesteps - 1))).astype(np.uint8)
        II = (t_step,) + II
        spike_times = np.zeros((self.nb_timesteps, img.shape[0], img.shape[1]), dtype=np.uint8)
        spike_times[II] = 1
        spike_times = spike_times.reshape(self.nb_timesteps, 1, img.shape[0], img.shape[1])
        return spike_times

    def __len__(self):
        return len(self.data)
    
    def data_shape(self, idx=0):
        img = self.data[idx]
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0).shape

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        spike_encoded_img = self.spike_encoding(img)
        return torch.tensor(spike_encoded_img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def load_mnist():
    mndata = MNIST('C:\\Users\\user\\python\\SpikingConvNet-main\\SpikingConvNet-main\\csnn\\data\\MNIST\\raw')  # Update the path to MNIST dataset
    # mndata = MNIST()
    images, labels = mndata.load_training()
    X_train, y_train = np.asarray(images), np.asarray(labels)
    X_train = X_train.reshape(-1, 28, 28)

    images, labels = mndata.load_testing()
    X_test, y_test = np.asarray(images), np.asarray(labels)
    X_test = X_test.reshape(-1, 28, 28)

    return X_train, y_train, X_test, y_test

def Mnist_dataloader(train_batch_size, test_batch_size, nb_timesteps):
    X_train, y_train, X_test, y_test = load_mnist()

    train_dataset = MNISTSpikeDataset(X_train, y_train, nb_timesteps)
    test_dataset = MNISTSpikeDataset(X_test, y_test, nb_timesteps)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    train_loader_for_test = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False)
    
    image_shape = train_dataset.data_shape()
    train_dataset_len = train_dataset.__len__()
    test_dataset_len = test_dataset.__len__()

    return train_loader, test_loader, train_loader_for_test, image_shape, train_dataset_len, test_dataset_len, y_train, y_test
