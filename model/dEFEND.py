from torch import nn
import torch
import torch.utils.data
from .attention_layer import AttentionLayer
from .co_attention_layer import CoAttentionLayer
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np
import itertools
from .sentencizer import Sentencizer
from .tokenizer import Tokenizer

class Defend(nn.Module):
    """
    PyTorch implementation of the dEFEND model.
    Original paper can be found at: https://dl.acm.org/doi/10.1145/3292500.3330935
    Kai Shu, Limeng Cui, Suhang Wang, Dongwon Lee, and Huan Liu. 2019.
     DEFEND: Explainable Fake News Detection.
     In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining (KDD '19).
     Association for Computing Machinery, New York, NY, USA, 395–405.
     https://doi.org/10.1145/3292500.3330935
    """

    def move_to_device(self, net):
        """
        Moves the model to the GPU, if available.
        :param net: The model / network to move
        :return: The model / network moved to the GPU
        """
        if len(self.opt.gpu_ids) > 0:
            net = net.to(self.opt.gpu_ids[0])
            net = nn.parallel.DataParallel(net, device_ids=self.opt.gpu_ids)
        return net

    def __init__(self, opt):
        """
        Constructor for the dEFEND model.
        Builds the model, initializes the embedding layers, and sets up the optimizer.
        :param opt: The options for the model
        """
        super(Defend, self).__init__()
        self.opt = opt
        encoding_dim = 2 * opt.d if opt.bidirectional else opt.d
        self.embedding_index = {"padding": torch.zeros(opt.embedding_dim)}
        self.embedding_mapping = {"padding": 0}
        self.sentencizer = Sentencizer()
        self.tokenizer = Tokenizer()
        self.metrics = [
            f1_score,
            recall_score,
            precision_score,
            roc_auc_score,
            accuracy_score
        ]

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
        self.fc = nn.Sequential(
            nn.Linear(2 * encoding_dim, 2),
            nn.Softmax(dim=-1)
        )

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
        self.loss = nn.CrossEntropyLoss()
        self.models = [
            self.article_embedding,
            self.comment_embedding,
            self.word_encoder,
            self.sentence_encoder,
            self.comment_encoder,
            self.co_attention,
            self.fc
        ]



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

        return encoded_texts

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


        return encoded_texts


    def forward(self, comments, articles, return_attention=False):
        """
        Forward pass of the model.
        :param comments: The comments
        :param articles: The articles
        :return:
        """
        # Convert input indexes to embeddings
        articles = self.article_embedding(articles)
        comments = self.comment_embedding(comments)

        # Encode the articles. The output is a tensor of shape (batch_size, max_sentence_count, 2 * d)
        article_sentence_encodings = torch.zeros((articles.shape[0], self.opt.max_sentence_count,
                                                  2 * self.opt.d if self.opt.bidirectional else self.opt.d))
        for i in range(articles.shape[0]):
            article = articles[i]
            article_word_embedding = self.word_encoder(article)
            article_sentence_embedding, _ = self.sentence_encoder(article_word_embedding)
            article_sentence_encodings[i] = article_sentence_embedding

        article_comments_encodings = torch.zeros((comments.shape[0], self.opt.max_comment_count,
                                                  2 * self.opt.d if self.opt.bidirectional else self.opt.d))
        for i in range(comments.shape[0]):
            comment = comments[i]
            comment_word_embedding = self.comment_encoder(comment)
            article_comments_encodings[i] = comment_word_embedding

        # Compute the co-attention output
        if return_attention:
            co_attention_output, As, Ac = self.co_attention((article_sentence_encodings, article_comments_encodings), return_attention=True)
            return self.fc(co_attention_output), As, Ac
        else:
            co_attention_output = self.co_attention((article_sentence_encodings, article_comments_encodings))
            # Feed the co-attention to the fully connected layer
            output = self.fc(co_attention_output)
            return output


    def backward(self, y_pred, y_true):
        """
        Backward pass of the model.
        :param y_pred: The predicted labels
        :param y_true: The true labels
        :return:
        """
        y_pred = y_pred.cpu()
        y_true = y_true.cpu()
        loss = self.loss(y_pred, y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def on_epoch_end(self, val_dataset, epoch):
        """
        Compute the metrics at the end of an epoch, and potentially save the model.
        :param articles_x_val:
        :param comments_x_val:
        :param y_val:
        :return:
        """
        for model in self.models:
            model.eval()
        y_pred_all = []
        y_true_all = []
        loss_vals = []
        print("Validation set")
        # Go through the validation dataset
        for batch in tqdm(val_dataset):
            articles, comments, y = batch
            y_pred = self.predict(articles, comments, require_index_conversion=False).cpu().detach()
            loss = self.loss(y_pred, y)
            loss_vals.append(loss.item())
            y_pred_all.append(y_pred.numpy())
            y_true_all.append(y.numpy())

        # Concatenate all predictions and true labels
        y_pred_all = np.concatenate(y_pred_all)
        y_true_all = np.concatenate(y_true_all)
        # Convert from one-hot to integer
        y_true_all = np.argmax(y_true_all, axis=-1)
        y_pred_all = np.argmax(y_pred_all, axis=-1)

        metrics = {}
        for metric in self.metrics:
            metrics[metric.__name__] = metric(y_true_all, y_pred_all)
        metrics['loss'] = np.mean(loss_vals)

        # Save the model if the epoch is a multiple of the save_epoch_freq
        if epoch % self.opt.save_epoch_freq == 0:
            # Check if the checkpoints directory exists. If not, create it
            import os
            if not os.path.exists(self.opt.checkpoints_dir):
                os.makedirs(self.opt.checkpoints_dir)
            torch.save(self.state_dict(), f'{self.opt.checkpoints_dir}/{self.opt.name}_{epoch}.pt')

        return metrics

    def predict(self, articles, comments, require_index_conversion=True, return_attention=False):
        if require_index_conversion:
            articles = self.to_embedding_indexes_articles(articles)
            comments = self.to_embedding_indexes_comments(comments)

        return self.forward(comments, articles, return_attention=return_attention)



    def fit(self, articles_x_train, comments_x_train, y_train, articles_x_val, comments_x_val, y_val, n_epochs):
        """
        Fit the model to the training data.
        :param articles_x_train:
        :param comments_x_train:
        :param y_train:
        :param articles_x_val:
        :param comments_x_val:
        :param y_val:
        :param n_epochs:
        :return:
        """

        # Convert the articles and comments to indexes
        articles_x_train = self.to_embedding_indexes_articles(articles_x_train)
        comments_x_train = self.to_embedding_indexes_comments(comments_x_train)
        articles_x_val = self.to_embedding_indexes_articles(articles_x_val)
        comments_x_val = self.to_embedding_indexes_comments(comments_x_val)

        # Convert all data to PyTorch datasets
        train_dataset = torch.utils.data.TensorDataset(articles_x_train, comments_x_train, torch.tensor(y_train, dtype=torch.float32))
        val_dataset = torch.utils.data.TensorDataset(articles_x_val, comments_x_val, torch.tensor(y_val, dtype=torch.float32))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.opt.batch_size, shuffle=False, num_workers=self.opt.num_workers)

        train_loss_vals = []
        metrics_vals = []
        for epoch in range(n_epochs):
            for model in self.models:
                model.train()
            print(f"Epoch: {epoch + 1}")
            train_loss_epoch = []
            for batch in tqdm(train_loader):
                articles, comments, y = batch
                y_pred = self.forward(comments, articles)
                loss = self.backward(y_pred, y)
                train_loss_epoch.append(loss)
            train_loss_vals.append(np.mean(train_loss_epoch))
            print(f"Epoch: {epoch + 1} - Training loss: {train_loss_vals[-1]}")
            metrics = self.on_epoch_end(val_loader, epoch)
            print(f"Epoch: {epoch + 1} - Validation metrics: {metrics}")
            metrics_vals.append(metrics)




