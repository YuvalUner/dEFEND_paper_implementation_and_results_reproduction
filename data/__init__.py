import os
import torch
import pandas as pd

def load_pytorch_dataset(opt):
    return torch.load(f"{opt.dataroot}/{opt.dataset_name}.pt")

def load_textual_articles(opt):
    dataroot = opt.dataroot
    fake = []
    real = []
    for file in os.listdir(dataroot + "/fake"):
        with open(dataroot + "/fake/" + file, 'r', encoding='utf-8') as f:
            fake.append(f.read())

    for file in os.listdir(dataroot + "/real"):
        with open(dataroot + "/real/" + file, 'r', encoding='utf-8') as f:
            real.append(f.read())

    true_labels = [[1, 0] for _ in range(len(real))] + [[0, 1] for _ in range(len(fake))]
    articles = real + fake
    return articles, true_labels

def load_articles_with_comments(opt):
    dataroot = opt.dataroot
    dataset_name = opt.dataset_name
    articles_file = pd.read_excel(dataroot + "/articles_with_comments.xlsx")
    comments_file = pd.read_excel(dataroot + "/comments.xlsx")
    # Select all articles with a type that starts with dataset name
    articles_file = articles_file[articles_file['id'].str.startswith(dataset_name)]

    fake = articles_file[articles_file['type'] == 0]
    real = articles_file[articles_file['type'] == 1]
    fake = fake['text'].to_list()
    fake = [str(i) for i in fake]
    real = real['text'].to_list()
    real = [str(i) for i in real]

    fake_comments_ids = articles_file['comments_ids'][articles_file['type'] == 0]
    real_comments_ids = articles_file['comments_ids'][articles_file['type'] == 1]
    fake_comments = []
    real_comments = []

    for i, comment_list in enumerate(fake_comments_ids):
        fake_comments.append([])
        comment_list = comment_list.split(' ')
        comment_list = [j.strip().replace("'", "").replace("[", "").replace("]", "") for j in comment_list if j != '']
        comments = comments_file[comments_file['id'].isin(comment_list)]['text'].to_list()
        fake_comments[i].extend(comments)

    for i, comment_list in enumerate(real_comments_ids):
        real_comments.append([])
        comment_list = comment_list.split(' ')
        comment_list = [j.strip().replace("'", "").replace("[", "").replace("]", "") for j in comment_list if j != '']
        comments = comments_file[comments_file['id'].isin(comment_list)]['text'].to_list()
        real_comments[i].extend(comments)


    true_labels = [[1, 0] for _ in range(len(real))] + [[0, 1] for _ in range(len(fake))]
    articles = real + fake
    comments = real_comments + fake_comments
    return articles, comments, true_labels
