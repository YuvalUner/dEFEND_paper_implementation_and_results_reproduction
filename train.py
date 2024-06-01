from options import TrainOptions
from model import Defend

if __name__ == '__main__':

    opt = TrainOptions().parse() # get training options
    defend = Defend(opt)
    text_example = ['This is a test. It is a great test. The best test the world has ever seen.', 'This is another test']
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
    y_train = [[1, 0], [0, 1]]
    articles_x_val = ['Val article 1', 'Val article 2']
    comments_x_val = [['Val comment 1', 'Val comment 2'], ['Val comment 3', 'Val comment 4']]
    y_val = [[1, 0], [0, 1]]

    defend.fit(text_example, comments_example, y_train, articles_x_val, comments_x_val, y_val, opt.max_epochs)


