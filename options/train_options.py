from .base_options import BaseOptions

class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
        parser.add_argument('--RMSprop_ro_param', type=float, default=0.9, help='RMSprop optimizer rho parameter')
        parser.add_argument('--RMSprop_eps', type=float, default=1e-08, help='RMSprop optimizer epsilon parameter')
        parser.add_argument('--RMSprop_decay', type=float, default=0.0, help='RMSprop optimizer decay parameter')
        parser.add_argument('--max_epochs', type=int, default=20, help='number of epochs to train')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='directory to save models')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving models')

        return parser