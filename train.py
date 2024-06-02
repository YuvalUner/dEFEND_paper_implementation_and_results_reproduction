from options import TrainOptions
from model import Defend
import torch
from data import load_articles
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    torch.set_warn_always(False)
    import warnings
    warnings.filterwarnings("ignore")
    # get training options
    opt = TrainOptions().parse()
    # Get the articles and their true labels
    articles, true_labels = load_articles(opt)
    x_train, x_val, y_train, y_val = train_test_split(articles, true_labels, test_size=0.2, random_state=42)
    # Fake, empty comments, because we can't get the real comments.
    # Thanks Elon Musk, those twitter API changes and 5000$ price tag are wonderful.
    train_comments = [[]] * len(x_train)
    val_comments = [[]] * len(x_val)

    defend = Defend(opt)
    # defend.fit(x_train, train_comments, y_train, x_val, val_comments, y_val, opt.max_epochs)
    pred, sent, com = defend.predict_explain(x_val[1], [[]])
    print(pred)
    print(sent)
    print(com)


