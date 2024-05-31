import torch
from torch import nn

class CoAttentionLayer(nn.Module):

    def __init__(self, opt, **kwargs):
        super(CoAttentionLayer, self).__init__()
        self.attention_dim = opt.k
        self.latent_dim = 2 * opt.d if opt.bidirectional else opt.d
        self.build()

    def build(self):
        # WL - first weight matrix, used for computing affinity matrix F
        self.Wl = nn.Parameter(torch.randn(self.latent_dim, self.latent_dim))

        # Ws and Wc - used for computing attention maps for sentences and comments
        self.Ws = nn.Parameter(torch.randn(self.attention_dim, self.latent_dim))
        self.Wc = nn.Parameter(torch.randn(self.attention_dim, self.latent_dim))

        # Whs and Whc - used for computing the attention weights vectors for sentences and comments
        self.Whs = nn.Parameter(torch.randn(1, self.attention_dim))
        self.Whc = nn.Parameter(torch.randn(1, self.attention_dim))


    def compute_mask(self, inputs, mask=None):
        return mask

    def forward(self, x):
        comment_rep = x[0]
        sentence_rep = x[1]

        comment_rep_transpose = comment_rep.transpose(-2, -1)
        sentence_rep_transpose = sentence_rep.transpose(-2, -1)

        # Compute affinity matrix F
        F = nn.Tanh()(torch.matmul(torch.matmul(comment_rep, self.Wl), sentence_rep_transpose))

        # Compute heatmaps Hs and Hc
        Hs = nn.Tanh()(torch.matmul(self.Ws, sentence_rep_transpose) + torch.matmul(torch.matmul(self.Wc, comment_rep_transpose), F))
        Hc = nn.Tanh()(torch.matmul(self.Wc, comment_rep_transpose) + torch.matmul(torch.matmul(self.Ws, sentence_rep_transpose), F.transpose(-1, -2)))

        # Compute attention weights vectors for sentences and comments
        As = torch.softmax(torch.matmul(self.Whs, Hs), dim=-1)
        Ac = torch.softmax(torch.matmul(self.Whc, Hc), dim=-1)

        # Compute the weighted representations
        weighted_sentence = sentence_rep_transpose * As
        weighted_comment = comment_rep_transpose * Ac

        weighted_sentence = torch.sum(weighted_sentence, dim=-1)
        weighted_comment = torch.sum(weighted_comment, dim=-1)

        return torch.concat((weighted_sentence, weighted_comment), dim=1)

