import argparse
import os
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')
        parser.add_argument('--batch_size', type=int, default=30, help='input batch size')
        parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
        parser.add_argument('--max_sentence_len', type=int, default=120, help='maximum sentence length. Sentences will be padded to this length if shorter')
        parser.add_argument('--max_sentence_count', type=int, default=50, help='maximum sentence count. More sentences will be ignored')
        parser.add_argument('--max_comment_count', type=int, default=50, help='maximum comment count per article. More comments will be ignored')
        parser.add_argument('--max_comment_len', type=int, default=120, help='maximum comment length. Comments will be padded to this length if shorter')
        parser.add_argument('--embedding_path', type=str, default=None, help='path to pretrained embeddings. If None, embeddings will be trained from scratch')
        parser.add_argument('--embedding_dim', type=int, default=100, help='dimension of word embeddings')
        parser.add_argument('--vocab_size', type=int, default=20000, help='maximum number of words in the vocabulary')
        parser.add_argument('--d', type=int, default=100, help='dimension of hidden layers and embeddings')
        parser.add_argument('--k', type=int, default=80, help='dimension of the attention layer')
        parser.add_argument('--name', type=str, default='model', help='name of the model')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--bidirectional', type=str2bool, default=True, help='use bidirectional GRU')
        parser.add_argument('--use_comments', type=str2bool, default=True, help='use comments as input. If False, only article text will be used')
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
