from options import BaseOptions
from model.attention_layer import AttentionLayer
import torch

if __name__ == '__main__':
    opt = BaseOptions().parse()   # get training options
    att_layer = AttentionLayer(k=10, input_shape=(10, 10))
    # Simulate a batch of 10 samples, each with an embedding dim of 10, and a sequence length of 30
    rand_input = torch.randn(10, 30, 10)
    output = att_layer(rand_input)