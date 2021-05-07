import os
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"


import argparse
import time
from os.path import join, dirname, abspath

from nntime import export_timings, set_global_sync, set_global_level, time_this
set_global_sync(True)

import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint #, wrap_fp16_model
from mmdet.models.backbones.repvgg import repvgg_det_model_convert

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector


def print_stats(var, name='', fmt='%.3g', cvt=lambda x: x, file=None):
    var = np.asarray(var)
    
    if name:
        prefix = name + ': '
    else:
        prefix = ''

    if len(var) == 1:
        print(('%sscalar: ' + fmt) % (
            prefix,
            cvt(var[0]),
        ), file=file)
    else:
        fmt_str = 'mean: %s; std: %s; min: %s; max: %s' % (
            fmt, fmt, fmt, fmt
        )
        print(('%s' + fmt_str) % (
            prefix,
            cvt(var.mean()),
            cvt(var.std(ddof=1)),
            cvt(var.min()),
            cvt(var.max()),
        ), file=file)

def wrap_fp16_model(model):
    """Wrap the FP32 model to FP16.

    1. Convert FP32 model to FP16.
    -- Remain some necessary layers to be FP32, e.g., normalization layers.

    Args:
        model (nn.Module): Model in FP32.
    """
    # convert model to fp16
    model.half()
    # patch the normalization layers to make it work in fp32 mode
    # patch_norm_fp32(model)
    # set `fp16_enabled` flag
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    # parser.add_argument('--data-dir')
    # parser.add_argument('--ssd-dir')
    # parser.add_argument('--timing-path')
    parser.add_argument('--n-sample', default=2000)
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--eval-out', help='file to write evaluation output')
    parser.add_argument(
        '--convert-repvgg', 
        action='store_true', 
        help='convert repvgg model')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument('--img_size', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if isinstance(cfg.data.test, dict):
    # specify manual img_size
        if args.img_size != 0:
            print("Scaling images to ", args.img_size)
            test_pipeline = cfg.data.test.pipeline
            for d in test_pipeline:
                if 'img_scale' in d:
                    d['img_scale'] = (args.img_size, args.img_size)
            cfg.data.test.pipeline = test_pipeline

    # modify path
    # for split in ['train', 'val', 'test']:
    #     if split in cfg.data:
    #         for key in ['root', 'img_prefix']:
    #             if key in cfg.data[split]:
    #                 cfg.data[split][key] = join(args.ssd_dir, cfg.data[split][key])
    #         for key in ['ann_file']:
    #             if key in cfg.data[split]:
    #                 cfg.data[split][key] = join(args.data_dir, cfg.data[split][key])

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.convert_repvgg:
        print("Converting repvgg model")
        cfg.model.backbone['deploy'] = True
        deploy_model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        model = repvgg_det_model_convert(model, deploy_model)
        
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # print(model)

    # @time_this()
    # def forward(self, *inputs, **kwargs):
    #     """Override the original forward function.

    #     The main difference lies in the CPU inference where the datas in
    #     :class:`DataContainers` will still be gathered.
    #     """
    #     if not self.device_ids:
    #         # We add the following line thus the module could gather and
    #         # convert data containers as those in GPU inference
    #         inputs, kwargs = self.scatter(inputs, kwargs, [-1])
    #         return self.module(*inputs[0], **kwargs[0])
    #     else:
    #         return super(MMDataParallel, self).forward(*inputs, **kwargs)
    # MMDataParallel.forward = forward
    # model = MMDataParallel(model, device_ids=[0])
    model = model.to('cuda')

    def process_dc(data):
        n = len(data['img_metas'])
        for i in range(n):
            data['img_metas'][i] = data['img_metas'][i].data[0]
        return data

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    model_time = []
    encoding_time = []

    results = []
    for i, data in enumerate(data_loader):
        data = process_dc(data)
        data['img'][0] = data['img'][0].to('cuda')

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        model_time.append(t2 - t1)

        if i >= num_warmup:
            if (i + 1) % args.log_interval == 0:
                print(f'Done image [{i + 1:<3}/ {args.n_sample}], runtime: {1e3*(sum(model_time[num_warmup:])/(len(model_time) - num_warmup)):.6g}ms, fps: {(i + 1 - num_warmup)/ sum(model_time[num_warmup:])} img / s')
        if (i + 1) == args.n_sample:
            fps = (i + 1 - num_warmup) / sum(model_time[num_warmup:])
            print(f'Overall fps: {fps:.1f} img / s')
            break

    s2ms = lambda x: 1e3*x
    print_stats(model_time[num_warmup:], 'Model time (ms)', cvt=s2ms)

    if args.eval_out:
        with open(args.eval_out, "a") as eval_out:
            print("\n")
            print(args.config, "img_size=", args.img_size if args.img_size != 0 else "1333,800", file=eval_out)
            print(f'fps: {args.n_sample/sum(model_time[num_warmup:])}', file=eval_out)
            print_stats(model_time[num_warmup:], 'Model time (ms)', cvt=s2ms, file=eval_out)
            print("\n", file=eval_out)

    # export_timings(model, args.timing_path, warmup=num_warmup)
    del results

if __name__ == '__main__':
    main()
