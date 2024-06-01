from options import TrainOptions
from model import Defend

if __name__ == '__main__':

    opt = TrainOptions().parse() # get training options
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
    text_example = text_example * 3000
    comments_example = comments_example * 3000
    y_train = [[1, 0], [0, 1]]
    y_train = y_train * 3000
    articles_x_val = ['Val article 1', 'Val article 2']
    articles_x_val = articles_x_val * 30
    comments_x_val = [['Val comment 1', 'Val comment 2'], ['Val comment 3', 'Val comment 4']]
    comments_x_val = comments_x_val * 30
    y_val = [[1, 0], [0, 1]]
    y_val = y_val * 30

    defend = Defend(opt)
    # defend.fit(text_example, comments_example, y_train, articles_x_val, comments_x_val, y_val, opt.max_epochs)
    pred, sent, com = defend.predict_explain(text_example[0], comments_example[0])
    print(pred)
    print(sent)
    print(com)


