from torch import nn
import torch

class Defend(nn.Module):

    def __init__(self, opt):
        super(Defend, self).__init__()
        self.opt = opt
        self.