from .dEFEND import Defend
import torch

class DefendNoComments(Defend):
    """
    This class is a subclass of the Defend class.
    The only difference is that this class does not use comments as input.
    It replaces the comments with the identity matrix.
    """

    def __init__(self, opt):
        super().__init__(opt)

    def forward(self, comments, articles, return_attention=False):

        articles = self.article_embedding(articles)

        article_sentence_encodings = torch.zeros((articles.shape[0], self.opt.max_sentence_count,
                                                  2 * self.opt.d if self.opt.bidirectional else self.opt.d))
        for i in range(articles.shape[0]):
            article = articles[i]
            article_word_embedding = self.word_encoder(article)
            article_sentence_embedding, _ = self.sentence_encoder(article_word_embedding)
            article_sentence_encodings[i] = article_sentence_embedding

        # Create a tensor of shape (batch_size, max_comment_count, embedding_dim), that would not affect the output
        comments = torch.eye(self.opt.max_comment_count, self.opt.d * 2 if self.opt.bidirectional else self.opt.d)
        comments = comments.unsqueeze(0).repeat(articles.shape[0], 1, 1)
        comments = comments.to(article_sentence_encodings.device)

        if return_attention:
            co_attention_output, As, Ac = self.co_attention((article_sentence_encodings, comments), return_attention=True)
            return self.fc(co_attention_output), As, Ac
        else:
            co_attention_output = self.co_attention((article_sentence_encodings, comments))
            # Feed the co-attention to the fully connected layer
            output = self.fc(co_attention_output)
            return output
