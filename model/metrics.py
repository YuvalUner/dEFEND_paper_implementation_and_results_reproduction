import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score


class Metrics:
    def __init__(self, platform):
        self.log_file = open('./Log_Defend_' + platform + '.txt', 'a')

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_auc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict_onehot = (
            np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]]))).round()
        val_targ_onehot = self.validation_data[2]
        val_predict = np.argmax(val_predict_onehot, axis=1)
        val_targ = np.argmax(val_targ_onehot, axis=1)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_auc = roc_auc_score(val_targ, val_predict)
        _val_acc = accuracy_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_auc.append(_val_auc)
        self.val_acc.append(_val_acc)
        pr_string = f"Epoch: {epoch} - val_accuracy: {_val_acc} - val_precision: {_val_precision}" +\
                    f" - val_recall {_val_recall} val_f1: {_val_f1} auc: {_val_auc}"
        print (pr_string)
        self.log_file.write(pr_string + '\n')