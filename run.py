from configs.base import ParamManager, add_config_param
from data.base import DataManager
from models import method_map
from utils.functions import set_torch_seed, save_results, set_output_path

import argparse
import logging
import os
import datetime
import itertools
import warnings
import time
import torch


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--logger_name', type=str, default='Multimodal Intent Recognition', help="Logger name for multimodal intent recognition.")

    parser.add_argument("--data_path", type=str, default='sharing/Data', help="The input data dir. Should contain text, video and audio data for the task.")

    parser.add_argument('--dataset', type=str, default='MIntRec2.0', help="The selected dataset.")

    parser.add_argument('--video_data_path', type=str, default='video_data', help="The directory of the video data.")

    parser.add_argument('--video_feats_path', type=str, default='swin_roi.pkl', help="The directory of the video features.")

    parser.add_argument('--audio_data_path', type=str, default='audio_data', help="The directory of the audio data.")

    parser.add_argument('--audio_feats_path', type=str, default='wavlm_feats.pkl', help="The directory of the audio features.")

    parser.add_argument("--desc_path", type=str, default='data/description', help='Description path')

    parser.add_argument("--compre_desc_path", type=str, default='compre_desc', help='Description path')

    parser.add_argument("--desc_suffix", type=str, default='', help="Desc file suffix override, e.g. '.core.qwen.tsv'")

    parser.add_argument("--text_backbone", type=str, default='bert-large-uncased', help="which backbone to use for text modality")

    parser.add_argument('--method', type=str, default='cesf', help="which method to use.")

    parser.add_argument("--config_file_name", type=str, default='cesf_mintrec2.py', help="The name of the config file.")

    parser.add_argument('--seed', type=int, default=0, help="The selected seed.")

    parser.add_argument('--num_workers', type=int, default=8, help="The number of workers to load data.")

    parser.add_argument('--log_id', type=str, default=None, help="The index of each logging file.")

    parser.add_argument('--gpu_id', type=str, default='0', help="The selected gpu id.")

    parser.add_argument("--train", action="store_true", help="Whether to train the model.")

    parser.add_argument("--save_model", action="store_true", help="save trained-model for multimodal intent recognition.")

    parser.add_argument("--save_results", action="store_true", help="save final results for multimodal intent recognition.")

    parser.add_argument('--log_path', type=str, default='logs', help="Logger directory.")

    parser.add_argument('--results_path', type=str, default='results', help="The path to save results.")

    parser.add_argument("--output_path", type=str, default='outputs', help="The output directory where all train data will be written.")

    parser.add_argument("--model_path", type=str, default='models', help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument('--cache_path', type=str, default='cache', help="The caching directory for pre-trained models.")

    parser.add_argument("--results_file_name", type=str, default='results.csv', help="The file name of all the results.")

    parser.add_argument('--save_pred', type=bool, default=False, help="Whether to save predictions.")

    args = parser.parse_args()

    args.train = True

    return args


def set_logger(args):

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.logger_name = f"{args.method}_{args.dataset}_{args.seed}"
    args.log_id = f"{args.logger_name}_{time}"

    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(args.log_path, args.log_id + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def set_up(args):

    save_model_name = f"{args.method}_{args.dataset}_{args.text_backbone}_{args.seed}"

    args.pred_output_path, args.model_output_path = set_output_path(args, save_model_name)

    set_torch_seed(args.seed)

    return args


def work(args, data, logger, debug_args=None):

    set_torch_seed(args.seed)

    method_manager = method_map[args.method]
    method = method_manager(args, data)

    args.num_params = round(sum(p.numel() for p in method.model.parameters()) / 1e6, 2)

    logger.info('Intent Recognition begins...')

    if args.train:
        logger.info('Training begins...')
        torch.cuda.reset_peak_memory_stats()
        train_t0 = time.time()
        method._train(args)
        train_time = time.time() - train_t0
        logger.info('Training is finished...')
        args.gpu_mem = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        epochs = getattr(method, 'epochs_trained', args.num_train_epochs)
        args.train_time_epoch = round(train_time / max(epochs, 1), 2)

    logger.info('Testing begins...')
    test_t0 = time.time()
    outputs = method._test(args)
    test_time = time.time() - test_t0
    logger.info('Testing is finished...')

    args.il = round(len(data.mm_data['test']) / max(test_time, 1e-6), 1)

    logger.info('Intent recognition is finished...')

    if args.save_results:
        logger.info('Results are saved in %s', str(os.path.join(args.results_path, args.results_file_name)))
        save_results(args, outputs, debug_args=debug_args)


def run(args, data, logger):

    debug_args = {}
    for k, v in args.items():
        if isinstance(v, list):
            debug_args[k] = v

    for result in itertools.product(*debug_args.values()):
        for i, key in enumerate(debug_args.keys()):
            args[key] = result[i]

        logger.info("=" * 30 + " Params " + "=" * 30)
        for k in args.keys():
            logger.info(f"{k}: {args[k]}")
        logger.info("=" * 30 + " End Params " + "=" * 30)

        work(args, data, logger, debug_args)


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    args = parse_arguments()
    param = ParamManager(args)
    args = param.args
    args = add_config_param(args, args.config_file_name)
    args = set_up(args)

    data = DataManager(args)

    logger = set_logger(args)

    run(args, data, logger)
