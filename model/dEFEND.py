from torch import nn
import torch
from .attention_layer import AttentionLayer
from .co_attention_layer import CoAttentionLayer
from tqdm import tqdm
import numpy as np
import itertools

class Defend(nn.Module):

    def move_to_device(self, net):
        if len(self.opt.gpu_ids) > 0:
            net = net.to(self.opt.gpu_ids[0])
            net = nn.parallel.DistributedDataParallel(net, device_ids=self.opt.gpu_ids)
        return net

    def __init__(self, opt):
        super(Defend, self).__init__()
        self.opt = opt
        encoding_dim = 2 * opt.d if opt.bidirectional else opt.d
        self.embedding_index = {}
        self.embedding_mapping = {}

        if opt.embedding_path is not None:
            with open(opt.embedding_path, 'r', encoding='utf-8') as f:
                print('Loading GloVe embeddings')
                for line in tqdm(f):
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    coefs = torch.from_numpy(coefs)
                    self.embedding_index[word] = coefs
                    self.embedding_mapping[word] = len(self.embedding_mapping)
            f.close()
            embedding_matrix = torch.zeros((len(self.embedding_index) + 1, opt.embedding_dim))
            print('Creating embedding matrix')
            for i, (word, coefs) in tqdm(enumerate(self.embedding_index.items())):
                embedding_matrix[i] = coefs
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(len(opt.word_index) + 1, opt.embedding_dim)



        # Word encoder - GRU followed by attention layer. Embedding dim is the input size
        # Encodes each sentence by computing the weighted sum of the contextualized word embeddings
        self.word_encoder = nn.Sequential(
            nn.GRU(opt.embedding_dim, opt.d, bidirectional=opt.bidirectional, batch_first=True),
            AttentionLayer(opt)
        )
        # Sentence encoder - GRU to give a contextualized sentence embeddings for each sentence, by the rest of
        # the sentences in the article
        self.sentence_encoder = nn.GRU(encoding_dim, opt.d, bidirectional=opt.bidirectional, batch_first=True)

        # Comment encoder - GRU followed by attention layer. Embedding dim is the input size
        # Encodes each comment by the weighted sum of its contextualized word embeddings
        self.comment_encoder = nn.Sequential(
            nn.GRU(opt.embedding_dim, opt.d, bidirectional=opt.bidirectional, batch_first=True),
            AttentionLayer(opt)
        )

        # Co-attention layer - computes the weighted sum of the sentence and comment embeddings
        self.co_attention = CoAttentionLayer(opt)

        # Fully connected layer, to predict if an article is fake or real
        self.fc = nn.Linear(2 * encoding_dim, 2)
        self.optimizer = torch.optim.RMSprop(
            itertools.chain(self.embedding.parameters(), self.word_encoder.parameters(), self.sentence_encoder.parameters(),
                            self.comment_encoder.parameters(), self.co_attention.parameters(), self.fc.parameters()),
            lr=opt.lr, alpha=opt.RMSprop_ro_param, eps=opt.RMSprop_eps, weight_decay=opt.RMSprop_decay
        )

        # If GPU is available, move all models to GPU. Otherwise, they will stay on the CPU
        self.word_encoder = self.move_to_device(self.word_encoder)
        self.sentence_encoder = self.move_to_device(self.sentence_encoder)
        self.comment_encoder = self.move_to_device(self.comment_encoder)
        self.co_attention = self.move_to_device(self.co_attention)
        self.fc = self.move_to_device(self.fc)


