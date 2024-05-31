from options import TrainOptions
from model.co_attention_layer import CoAttentionLayer
import torch
from model import Defend

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    defend = Defend(opt)
    text_example = ['This is a test. It is a great test. The best test the world has ever seen.', 'This is another test']
    indexes = defend.to_embedding_indexes_articles(text_example)