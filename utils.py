import torch
import numpy as np
from torchvision.datasets import MNIST
from CONSTANTS import TRAIN_BATCH_SIZE

def Latency_coding(num_samples, image_shape, dataloader):
    """
    "0" pixel still "0"
    Other pixel transformed by (1-pixel)
    (All pixels consisted by 0~1 values)
    """
    transformed_dataset = torch.zeros((num_samples, *image_shape), dtype=torch.float32)
    for idx, (images, _) in enumerate(dataloader):
        transformed_image = (torch.where(images == 0, images, 1 - images))
        transformed_dataset[idx] = transformed_image
    return transformed_dataset


def change_conv_sizes(csnn: classmethod , layer_num: int, in_channel: int, crop_size: int, mode: str, device:torch.device) -> None :
    if mode == "pass" :
        csnn.conv_layers[layer_num].recorded_spks = torch.zeros((in_channel,
                                                         int(crop_size)+2*csnn.conv_layers[0].padding[0], 
                                                         int(crop_size)+2*csnn.conv_layers[0].padding[1]))
        out_height = int(((crop_size + 2 * csnn.conv_layers[layer_num].padding[0] - csnn.conv_layers[layer_num].kernel_size[0]) / csnn.conv_layers[layer_num].stride[0]) + 1)
        out_width = int(((crop_size + 2 * csnn.conv_layers[layer_num].padding[1] - csnn.conv_layers[layer_num].kernel_size[1]) / csnn.conv_layers[layer_num].stride[1]) + 1)
        csnn.conv_layers[layer_num].pot = torch.zeros((TRAIN_BATCH_SIZE, csnn.conv_layers[layer_num].out_channels, int(out_height), int(out_width)))

    elif mode == "train" :
        csnn.conv_layers[layer_num].recorded_spks = torch.zeros((TRAIN_BATCH_SIZE, in_channel, int(crop_size), int(crop_size)))
        csnn.conv_layers[layer_num].pot = torch.zeros((TRAIN_BATCH_SIZE, csnn.conv_layers[layer_num].out_channels, 1, 1))

    csnn.conv_layers[layer_num].active_neurons = torch.ones(csnn.conv_layers[layer_num].pot.shape).to(torch.bool)
    csnn.conv_layers[layer_num].stdp_neurons = torch.ones(csnn.conv_layers[layer_num].pot.shape).to(torch.bool)
    csnn.conv_layers[layer_num].output_shape = csnn.conv_layers[layer_num].pot.shape
    return


def get_sorted_nonzero_indices(x: torch.tensor) -> torch.tensor :
    non_zero_mask = x > 0
    non_zero_values = x[non_zero_mask]
    sorted_indices = torch.argsort(non_zero_values)
    non_zero_indices = torch.nonzero(x > 0, as_tuple=False)
    return non_zero_indices[sorted_indices]


def get_nonzero_softmax(tensor: torch.tensor, dim: int = 0) -> torch.tensor :
    mask = tensor != 0

    # Replace zeros with a very negative value to exclude them from softmax
    adjusted_pot = tensor.clone()
    adjusted_pot[~mask] = float('-inf')

    # Perform softmax along dim=0, ignoring -inf values
    soft_wta_pot = torch.softmax(adjusted_pot, dim=dim)

    # Ensure the values corresponding to the zeros are also zero in the result
    soft_wta_pot[~mask] = 0

    return soft_wta_pot.to(tensor.device)