from options import TrainOptions
from model import Defend, DefendNoComments
import torch
from data import load_pytorch_dataset
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    torch.set_warn_always(False)
    import warnings
    warnings.filterwarnings("ignore")
    # get training options
    opt = TrainOptions().parse()
    # Get the articles and their true labels
    dataset = load_pytorch_dataset(opt)
    articles = dataset[:][0]
    true_labels = dataset[:][2]
    comments = dataset[:][1]
    x_train, x_val, train_comments, val_comments, y_train, y_val = train_test_split(articles, comments, true_labels, test_size=0.2, random_state=42)
    # # Fake, empty comments, because we can't get the real comments.
    # # Thanks Elon Musk, those twitter API changes and 5000$ price tag are wonderful.
    # train_comments = [[]] * len(x_train)
    # val_comments = [[]] * len(x_val)

    if opt.use_comments:
        defend = Defend(opt)
    else:
        defend = DefendNoComments(opt)
    train_history, val_history = defend.fit(x_train, train_comments, y_train, x_val, val_comments, y_val, opt.max_epochs, require_index_conversion=False)


