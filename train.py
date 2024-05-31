from options import TrainOptions
from model.co_attention_layer import CoAttentionLayer
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import Defend

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    defend = Defend(opt)
    text_example = ['This is a test. It is a great test. The best test the world has ever seen.', 'This is another test']
    sentences_dataset = defend.to_embedding_indexes_articles(text_example)
    comments_example = [
        [
            'This is a comment. It is a great comment. The best comment the world has ever seen.',
            'This is another comment'
        ],
        [
            'This is a comment. It is a great comment. The best comment the world has ever seen.',
            'This is another comment'
        ]
    ]
    comments_dataset = defend.to_embedding_indexes_comments(comments_example)

    dataset = TensorDataset(sentences_dataset, comments_dataset)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    for batch in dataloader:
        sentences, comments = batch
        output = defend.forward(comments, sentences)
        print(output)
        break