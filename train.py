from options import BaseOptions
from model.co_attention_layer import CoAttentionLayer
import torch
from model import Defend

if __name__ == '__main__':
    opt = BaseOptions().parse()   # get training options
    defend = Defend(opt)