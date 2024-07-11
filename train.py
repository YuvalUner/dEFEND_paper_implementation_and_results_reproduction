from options import TrainOptions
from model import Defend, DefendNoComments
import torch
from data import load_pytorch_dataset, load_articles_with_comments
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    torch.set_warn_always(False)
    import warnings
    warnings.filterwarnings("ignore")
    # get training options
    opt = TrainOptions().parse()
    # Get the articles and their true labels
    if not opt.require_preprocessing:
        dataset = load_pytorch_dataset(opt)
        articles = dataset[:][0]
        true_labels = dataset[:][2]
        comments = dataset[:][1]
    else:
        articles, comments, true_labels = load_articles_with_comments(opt)
    x_train, x_val, train_comments, val_comments, y_train, y_val = train_test_split(articles, comments, true_labels, test_size=0.2, random_state=42)

    if opt.use_comments:
        defend = Defend(opt)
    else:
        defend = DefendNoComments(opt)
    train_history, val_history = defend.fit(x_train, train_comments, y_train, x_val, val_comments, y_val, opt.max_epochs,
                                            require_index_conversion=opt.require_preprocessing)
    # Save the model
    torch.save(defend.state_dict(), f"{opt.checkpoints_dir}/{opt.name}.pt")


