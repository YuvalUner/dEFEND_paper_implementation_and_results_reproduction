import os


def load_articles(opt):
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