from options import BaseOptions
from model.co_attention_layer import CoAttentionLayer
import torch

if __name__ == '__main__':
    opt = BaseOptions().parse()   # get training options
    att_layer = CoAttentionLayer(opt)
    # Simulate a batch of 10 samples, each with an embedding dim of 10, and a sequence length of 30
    rand_input_1 = torch.randn(30, 30, 200)
    rand_input_2 = torch.randn(30, 20, 200)
    rand_input = [rand_input_1, rand_input_2]
    output = att_layer(rand_input)