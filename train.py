from options import TrainOptions
from model import Defend
import torch
from data import load_articles
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    torch.set_warn_always(False)
    import warnings
    warnings.filterwarnings("ignore")
    opt = TrainOptions().parse() # get training options
    articles, true_labels = load_articles(opt)
    x_train, x_val, y_train, y_val = train_test_split(articles, true_labels, test_size=0.2, random_state=42)
    train_comments = [[]] * len(x_train)
    val_comments = [[]] * len(x_val)

    defend = Defend(opt)
    defend.fit(x_train, train_comments, y_train, x_val, val_comments, y_val, opt.max_epochs)
    pred, sent, com = defend.predict_explain(x_val[1], [[]])
    print(pred)
    print(sent)
    print(com)


