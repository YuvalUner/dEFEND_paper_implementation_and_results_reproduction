from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch
import torch.utils.data
from .attention_layer import AttentionLayer
from .co_attention_layer import CoAttentionLayer
from tqdm import tqdm
import numpy as np
import itertools
from .sentencizer import Sentencizer
from .tokenizer import Tokenizer

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
        self.embedding_index = {"padding": torch.zeros(opt.embedding_dim)}
        self.embedding_mapping = {"padding": 0}
        self.sentencizer = Sentencizer()
        self.tokenizer = Tokenizer()

        if opt.embedding_path is not None:
            # Load the GloVe embeddings, if provided
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
            # Different embedding layers for articles and comments, as defined in the original code.
            # Likely due to the fact that articles and comments have different probability distributions,
            # and thus different embeddings are needed, despite the fact that the same GloVe embeddings are used as a basis
            self.article_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
            self.comment_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        # If not using GloVe embeddings, create random embeddings and train them from scratch during training
        else:
            self.article_embedding = nn.Embedding(opt.vocab_size + 1, opt.embedding_dim)
            self.comment_embedding = nn.Embedding(opt.vocab_size + 1, opt.embedding_dim)



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

        # If GPU is available, move all models to GPU. Otherwise, they will stay on the CPU
        self.article_embedding = self.move_to_device(self.article_embedding)
        self.comment_embedding = self.move_to_device(self.comment_embedding)
        self.word_encoder = self.move_to_device(self.word_encoder)
        self.sentence_encoder = self.move_to_device(self.sentence_encoder)
        self.comment_encoder = self.move_to_device(self.comment_encoder)
        self.co_attention = self.move_to_device(self.co_attention)
        self.fc = self.move_to_device(self.fc)

        self.optimizer = torch.optim.RMSprop(
            itertools.chain(self.article_embedding.parameters(), self.comment_embedding.parameters(),
                            self.word_encoder.parameters(), self.sentence_encoder.parameters(),
                            self.comment_encoder.parameters(), self.co_attention.parameters(), self.fc.parameters()),
            lr=opt.lr, alpha=opt.RMSprop_ro_param, eps=opt.RMSprop_eps, weight_decay=opt.RMSprop_decay
        )



    def to_embedding_indexes_articles(self, articles):
        """
        Converts the articles to indexes.
        The main differences between this function and the matching comment function are the usage of
        the sentencizer to split the article into sentence, and the use of the opt.max_sentence_len and opt.max_sentence_count.
        :param articles:
        :return:
        """
        # Create a tensor to store the encoded articles
        encoded_texts = torch.zeros((len(articles), self.opt.max_sentence_count, self.opt.max_sentence_len), dtype=torch.int32)
        print("Converting articles to indexes")
        for i, article in tqdm(enumerate(articles)):
            # Lowercase the article, split it into sentences, tokenize the sentences, and convert the tokens to indexes
            article = article.lower()
            sentenceized_article = self.sentencizer(article)
            tokenized_sentences = self.tokenizer(sentenceized_article)
            indexed_sentences = [torch.tensor([self.embedding_mapping.get(word, 0) for word in sentence]) for sentence in tokenized_sentences]

            # Pad the article to the maximum sentence count
            if len(indexed_sentences) < self.opt.max_sentence_count:
                indexed_sentences += [torch.zeros(self.opt.max_sentence_len, dtype=torch.int32)] * (self.opt.max_sentence_count - len(indexed_sentences))
            else:
                indexed_sentences = indexed_sentences[:self.opt.max_sentence_count]

            # Convert the list of sentences to a tensor, and pad each sentence to the maximum sentence length
            for j, sentence in enumerate(indexed_sentences):
                if len(sentence) < self.opt.max_sentence_len:
                    sentence = torch.cat((sentence, torch.zeros(self.opt.max_sentence_len - len(sentence), dtype=torch.int32)))
                encoded_texts[i][j] = sentence

        dataset = torch.utils.data.TensorDataset(encoded_texts)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.opt.batch_size,
                                                 shuffle=False, num_workers=self.opt.num_workers)
        return dataloader

    def to_embedding_indexes_comments(self, comments):
        """
        Converts the comments to indexes.
        The main differences between this function and the matching article function are the usage of
        the opt.max_comment_len and the fact that the comments are not split into sentences.
        :param comments:
        :return:
        """
        # Create a tensor to store the encoded comments
        encoded_texts = torch.zeros((len(comments), self.opt.max_comment_count, self.opt.max_comment_len), dtype=torch.int32)
        print("Converting comments to indexes")
        for i, comment_collection in tqdm(enumerate(comments)):
            for j, comment in enumerate(comment_collection):
                if j >= self.opt.max_comment_count:
                    break
                # Lowercase the comment, tokenize it, and convert the tokens to indexes
                comment = comment.lower()
                tokenized_comment = self.tokenizer(comment)
                indexed_comment = torch.tensor([self.embedding_mapping.get(word, 0) for word in tokenized_comment])

                # Pad the comment to the maximum comment length
                if len(indexed_comment) < self.opt.max_comment_len:
                    indexed_comment = torch.cat((indexed_comment, torch.zeros(self.opt.max_comment_len - len(indexed_comment), dtype=torch.int32)))
                encoded_texts[i][j] = indexed_comment


        dataset = torch.utils.data.TensorDataset(encoded_texts)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.opt.batch_size,
                                                 shuffle=False, num_workers=self.opt.num_workers)
        return dataloader



