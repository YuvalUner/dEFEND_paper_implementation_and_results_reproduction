import torch
from torch import nn

class AttentionLayer(nn.Module):

    def __init__(self, opt, **kwargs):
        super(AttentionLayer, self).__init__()
        self.attention_dim = opt.k
        self.latent_dim = 2 * opt.d if opt.bidirectional else opt.d
        self.build()

    def build(self):
        self.W = nn.Parameter(torch.randn(self.latent_dim, self.attention_dim))
        self.b = nn.Parameter(torch.randn(self.attention_dim))
        self.u = nn.Parameter(torch.randn(self.attention_dim, 1))

    def compute_mask(self, inputs, mask=None):
        return mask

    def forward(self, x, mask=None):
        if isinstance(x, tuple):
            x = x[0]
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
