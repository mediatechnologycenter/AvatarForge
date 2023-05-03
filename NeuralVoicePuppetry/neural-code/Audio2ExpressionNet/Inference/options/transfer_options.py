from .base_options import BaseOptions


class TransferOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        #  Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        parser.add_argument('--write_no_images', action='store_true', help='compute validation')


        parser.add_argument('--source_dir', type=str, default='./datasets/', help='loads source files (expressions, audio, uvs).')

        parser.add_argument('--source_actor', type=str, default='', help='source actor directory')
        parser.add_argument('--target_actor', type=str, default='', help='target actor directory')

        parser.add_argument('--transfer_path', type=str, help='path to output the transfer files')
        parser.add_argument('--base_path', type = str, default= '../..', help='path for mappings folder')

        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser

class Audio2ExprOptions(TransferOptions):
    def initialize(self, parser):
        parser = TransferOptions.initialize(self, parser)
        parser.add_argument('--use_mapping', action='store_true', help='use mapping function.')
        parser.add_argument('--mapping_path', type=str, default='', help='path to mapping function.')
        parser.add_argument('--out_dir', type=str, default='', help='path to output directory.')
        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser