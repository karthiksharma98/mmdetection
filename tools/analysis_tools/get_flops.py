import argparse
import time
import torch
from mmcv import Config, DictAction

from mmdet.models import build_detector

from mmdet.models.backbones.repvgg import repvgg_det_model_convert, RepVGGConvModule
from mmdet.models.necks.dilated_encoder import Bottleneck
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default=None, help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 56, 56],
        help='input image size')
    parser.add_argument(
        '--convert-repvgg', 
        action='store_true', 
        help='convert repvgg model')
    # new cfg
    parser.add_argument(
        '--deploy', 
        default=False, 
        help='convert repvgg model')
    parser.add_argument(
        '--model', 
        help='model type')
    parser.add_argument(
        '--dilation',
        type=int, 
        default=1,
        help='dilation rate')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def speed(model, input_shape):    
    inputs = torch.randn(1, *input_shape).cuda()
    t0 = time.time()
    for i in range(10):
        model(inputs)
    t1 = time.time()
    return (t1 - t0) / 10 * 1000


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    elif len(args.shape) == 3:
        input_shape = (args.shape[0], args.shape[1], args.shape[2])
    else:
        raise ValueError('invalid input shape')

    if args.config:
        cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])

    if args.model == 'repvgg':
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

    elif args.model == 'repblock':
        model = RepVGGConvModule(
            in_channels=input_shape[0],
            out_channels=input_shape[0],
            kernel_size=3,
            stride=1,
            padding=args.dilation,
            dilation=args.dilation,
            groups=1,
            activation='ReLU',
            padding_mode='zeros',
            deploy=args.deploy
        )
    elif args.model == 'bottleneck':
        model = Bottleneck(
            in_channels=input_shape[0],
            mid_channels=512, # default setting according to yolof neck
            dilation=args.dilation,
        )
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(args.model))


    if args.convert_repvgg and args.model == 'repvgg':
        print("Converting repvgg model")
        cfg.model.backbone['deploy'] = True
        deploy_model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        model = repvgg_det_model_convert(model, deploy_model)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        # raise NotImplementedError(
        #     'FLOPs counter is currently not currently supported with {}'.
        #     format(model.__class__.__name__))
        pass

    flops, params = get_model_complexity_info(model, input_shape)
    time_usage = speed(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Model: {args.model}\nDeploy: {args.deploy}\nDilation: {args.dilation}\nFlops: {flops}\nParams: {params}\nTime: {time_usage} ms\n{split_line}\n')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
