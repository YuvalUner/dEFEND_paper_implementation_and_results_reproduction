import argparse
import os
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')
        parser.add_argument('--batch_size', type=int, default=30, help='input batch size')
        parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
        parser.add_argument('--max_sentence_len', type=int, default=120, help='maximum sentence length. Longer sentences will be truncated')
        parser.add_argument('--max_sentence_count', type=int, default=50, help='maximum sentence count. More sentences will be ignored')
        parser.add_argument('--max_comment_count', type=int, default=50, help='maximum comment count. More comments will be ignored')
        parser.add_argument('--max_comment_len', type=int, default=120, help='maximum comment length. Longer comments will be truncated')
        parser.add_argument('--embedding_path', type=str, default=None, help='path to pretrained embeddings. If None, embeddings will be trained from scratch')
        parser.add_argument('--embedding_dim', type=int, default=100, help='dimension of word embeddings')
        parser.add_argument('--max_epochs', type=int, default=20, help='number of epochs to train')
        parser.add_argument('--vocab_size', type=int, default=20000, help='maximum number of words in the vocabulary')
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
        parser.add_argument('--RMSprop_ro_param', type=float, default=0.9, help='RMSprop optimizer rho parameter')
        parser.add_argument('--RMSprop_eps', type=float, default=1e-08, help='RMSprop optimizer epsilon parameter')
        parser.add_argument('--RMSprop_decay', type=float, default=0.0, help='RMSprop optimizer decay parameter')
        parser.add_argument('--d', type=int, default=100, help='dimension of hidden layers and embeddings')
        parser.add_argument('--k', type=int, default=80, help='dimension of the attention layer')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='directory to save models')
        parser.add_argument('--name', type=str, default='model', help='name of the model')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--bidirectional', type=bool, default=True, help='use bidirectional GRU')

        self.initialized = True

        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

        return message

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        # process opt.suffix
        # if opt.suffix:
        #     suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #     opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
