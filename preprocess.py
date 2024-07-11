from model import Defend
from options import TrainOptions
import torch
from data import load_textual_articles, load_articles_with_comments

if __name__ == '__main__':
    opt = TrainOptions().parse()
    articles, comments, labels = load_articles_with_comments(opt)
    model = Defend(opt)
    article_indexes = model.to_embedding_indexes_articles(articles)
    comment_indexes = model.to_embedding_indexes_comments(comments)
    dataset = torch.utils.data.TensorDataset(article_indexes, comment_indexes, torch.tensor(labels, dtype=torch.float32))
    # Save the dataset
    torch.save(dataset, f"{opt.dataroot}/{opt.dataset_name}.pt")