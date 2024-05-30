import torch
from torch import nn

class AttentionLayer(nn.Module):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__()
        self.attention_dim = kwargs['k']
        self.build(kwargs['input_shape'])

    def build(self, input_shape):
        self.W = nn.Parameter((torch.randn(input_shape[-1], self.attention_dim)))
        self.b = nn.Parameter(torch.randn(self.attention_dim))
        self.u = nn.Parameter((torch.randn(self.attention_dim, 1)))
        self.trainable_variables = [self.W, self.b, self.u]

    def compute_mask(self, inputs, mask=None):
        return mask

    def forward(self, x, mask=None):
        u_it = torch.tanh(torch.matmul(x, self.W) + self.b)
        a_it = torch.matmul(u_it, self.u)
        a_it = torch.squeeze(a_it, -1)
        a_it = torch.softmax(a_it, dim=-1)
        if mask is not None:
            a_it = a_it * mask

        a_it = torch.unsqueeze(a_it, -1)
        weighted_input = x * a_it
        output = torch.sum(weighted_input, dim=1)

        return output
