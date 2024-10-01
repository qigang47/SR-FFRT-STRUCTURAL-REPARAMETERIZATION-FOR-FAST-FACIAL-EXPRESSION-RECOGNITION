#
# For acknowledgement see accompanying ACKNOWLEDGEMENTS file.
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
from sklearn.manifold import TSNE
import argparse
import os
import csv
import glob
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

import yaml
from timm.models.layers import apply_test_time_pool
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    RealLabelsImagenet,
)
from timm.utils import (
    accuracy,
    AverageMeter,
    natural_key,
    setup_default_logging,
    set_jit_legacy,
)


from models.modules.mobileone import reparameterize_model
from train import config_parser

has_apex = False
try:
    from apex import amp

    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("validate")


config_parser=parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")

parser.add_argument(
    '-c',
    '--config',
    default="E:\ml-fastvit-main\output\\train\\fer2013\\args.yaml",
    type=str,
    metavar='FILE',
    help='YAML config file specifying default arguments')
parser.add_argument(
    '--data_dir',
    metavar='DIR',
    default='',
    help='path to dataset')
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--split",
    metavar="NAME",
    default="validation",
    help="dataset split (default: validation)",
)
parser.add_argument(
    "--dataset-download",
    action="store_true",
    default=False,
    help="Allow download of dataset for torch/ and tfds/ datasets that support it.",
)
parser.add_argument(
    "--model",
    "-m",
    metavar="NAME",
    default="fastvit_t12",
    help="model architecture (default: dpn92)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=1,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 2)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--img-size",
    default=None,
    type=int,
    metavar="N",
    help="Input image dimension, uses model default if empty",
)
parser.add_argument(
    "--input-size",
    default=[1, 112, 112],
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 256 256), uses model default if empty",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop pct",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "--num-classes", type=int, default=8, help="Number classes in dataset"
)
parser.add_argument(
    "--class-map",
    default="",
    type=str,
    metavar="FILENAME",
    help='path to class to idx mapping file (default: "")',
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--log-freq",
    default=10,
    type=int,
    metavar="N",
    help="batch logging frequency (default: 10)",
)
parser.add_argument(
    "--checkpoint",
    default="E:\ml-fastvit-main\output\\train\\fer2013\model_best.pth.tar",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPUS to use")
parser.add_argument(
    "--test-pool", dest="test_pool", action="store_true", help="enable test time pool"
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.",
)
parser.add_argument(
    "--apex-amp",
    action="store_true",
    default=False,
    help="Use NVIDIA Apex AMP mixed precision",
)
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)
parser.add_argument(
    "--tf-preprocessing",
    action="store_true",
    default=False,
    help="Use Tensorflow preprocessing pipeline (require CPU TF installed",
)
parser.add_argument(
    "--use-ema",
    dest="use_ema",
    action="store_true",
    help="use ema version of weights if present",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)
parser.add_argument(
    "--legacy-jit",
    dest="legacy_jit",
    action="store_true",
    help="use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance",
)
parser.add_argument(
    "--results-file",
    default="",
    type=str,
    metavar="FILENAME",
    help="Output csv file for validation results (summary)",
)
parser.add_argument(
    "--real-labels",
    default="",
    type=str,
    metavar="FILENAME",
    help="Real labels JSON file for imagenet evaluation",
)
parser.add_argument(
    "--valid-labels",
    default="",
    type=str,
    metavar="FILENAME",
    help="Valid label indices txt file for validation of partial label space",
)
parser.add_argument(
    "--use-inference-mode",
    dest="use_inference_mode",
    action="store_true",
    default=False,
    help="use inference mode version of model definition.",
)


def visualize_tsne(tsne_results, labels):

  emotions =  ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

  label_color_map = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y', 6: 'k'}

  plt.figure(figsize=(10, 8))

  for i in range(len(emotions)):
    emotion = emotions[i]
    color = label_color_map[i]
    indices = np.where(np.array(labels) == i)

    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                color=color, label=emotion, s=50, alpha=0.7)

  plt.xlabel('Dimension 1')
  plt.ylabel('Dimension 2')
  plt.title('t-SNE Visualization of Facial Expressions')

  plt.legend(emotions, loc='best')

  plt.grid(True)
  plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    emotions =  ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
    abbreviations = [emotion[:2].upper() for emotion in emotions]
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm * 100  # 转换为百分比
        print("Normalized confusion matrix (in percentage)")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(10, 8))  # Increase the figure size
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=abbreviations, yticklabels=abbreviations,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    ax.tick_params(axis='both', which='major', labelsize=10)  # Increase tick label size

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if normalize:
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = format(cm[i, j], fmt)
                ax.text(j, i, f"{percentage}%",
                        ha="center", va="center", fontsize=9,  # Increase fontsize
                        color="white" if cm[i, j] > thresh else "black")
    else:
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center", fontsize=9,  # Increase fontsize
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    print(args)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
def validate(args):
    # might as well try to validate something
    args, args_text = _parse_args()
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    correct_per_class = torch.zeros(args.num_classes)
    total_per_class = torch.zeros(args.num_classes)
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
           _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info("Validating in mixed precision with native PyTorch AMP.")
    elif args.apex_amp:
        _logger.info("Validating in mixed precision with NVIDIA APEX AMP.")
    else:
        _logger.info("Validating in float32. AMP not enabled.")

    if args.legacy_jit:
        set_jit_legacy()
    model_names = list_models(args.model)
    correct_per_class = torch.zeros(args.num_classes, dtype=torch.float32).cuda()
    total_per_class = torch.zeros(args.num_classes, dtype=torch.float32).cuda()

    # create model
    model = create_model(
        args.model,
      #  pretrained=args.pretrained,
        num_classes=args.num_classes,
        global_pool=args.gp,
        scriptable=args.torchscript,

    )
    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    # Reparameterize model
    model.eval()
    if not args.use_inference_mode:
        _logger.info("Reparameterizing Model %s" % (args.model))
        model = reparameterize_model(model)
    setattr(model, "pretrained_cfg", model.__dict__["default_cfg"])

    data_config = resolve_data_config(
        vars(args), model=model, use_test_size=True, verbose=True
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(
            model, data_config, use_test_size=True
        )

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level="O1")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()
    print(args.data_dir)
    dataset = create_dataset(
        root=args.data_dir,
        name=args.dataset,
        split=args.split,
        download=False,
        load_bytes=args.tf_preprocessing,
        class_map=args.class_map,
        batch_size=args.batch_size
    )
      # 检查路径是否是一个目录

    if args.valid_labels:
        with open(args.valid_labels, "r") as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(
            dataset.filenames(basename=True), real_json=args.real_labels
        )
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config["crop_pct"]


    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing,
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_targets = []
    all_preds = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn(
            (args.batch_size,) + tuple(data_config["input_size"])
        ).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        model(input)
        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
             output,x = model(input)
             all_outputs.append(x)
            if valid_labels is not None:
                output = output[:, valid_labels]
            loss = criterion(output, target)
            _, preds = torch.max(output, 1)
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            if real_labels is not None:
                real_labels.add_result(output)
            pred = output.argmax(1)
            for i in range(args.num_classes):
                correct_per_class[i] += (pred == target)[target == i].sum()
                total_per_class[i] += (target == i).sum()
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if batch_idx % args.log_freq == 0:
                _logger.info(
                    "Test: [{0:>4d}/{1}]  "
                    "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  "
                    "Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})".format(
                        batch_idx,
                        len(loader),
                        batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )
    # 在 'with torch.no_grad():' 循环结束后
    plot_confusion_matrix(all_targets, all_preds, classes=[str(i) for i in range(args.num_classes)])
    all_outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
    print(all_outputs.shape)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_outputs)
    visualize_tsne(tsne_results, all_targets)
    print(set(all_targets))

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        top1=round(top1a, 4),
        top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4),
        top5_err=round(100 - top5a, 4),
        img_size=data_config["input_size"][-1],
        cropt_pct=crop_pct,
        interpolation=data_config["interpolation"],
    )
    accuracy_per_class = correct_per_class / total_per_class
    accuracy_per_class = accuracy_per_class.cpu().numpy()

    print("Accuracy for each class:")
    for i in range(args.num_classes):
        print(f"Class {i}: {accuracy_per_class[i] * 100:.2f}%")
    _logger.info(
        " * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})".format(
            results["top1"], results["top1_err"], results["top5"], results["top5_err"]
        )
    )

    return results


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + "/*.pth.tar")
        checkpoints += glob.glob(args.checkpoint + "/*.pth")
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == "all":
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(
                pretrained=True, exclude_filters=["*_in21k", "*_in22k"]
            )
            model_cfgs = [(n, "") for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, "") for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or "./results-all.csv"
        _logger.info(
            "Running bulk validation on these pretrained models: {}".format(
                ", ".join(model_names)
            )
        )
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print("Validating with batch size: %d" % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print(
                                "Validation failed with no ability to reduce batch size. Exiting."
                            )
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result["checkpoint"] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x["top1"], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate(args)


def write_results(results_file, results):
    with open(results_file, mode="w") as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == "__main__":
    main()
