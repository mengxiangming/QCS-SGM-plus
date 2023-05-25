import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
from runners import *

import os
import subprocess

def get_gpu_with_most_memory():
    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        best_gpu_id = memory_free_values.index(max(memory_free_values))

        if max(memory_free_values) < ACCEPTABLE_AVAILABLE_MEMORY:
            raise ValueError('No GPU with enough free memory available.')

        return best_gpu_id
    except Exception as e:
        print("Error:", e)
        return None


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default = "mnist.yml",  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--exp', type=str, default='./exp', help='Path for saving running related data.')
    parser.add_argument('--model_dir', type=str, default="mnist", help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--sample', action='store_false', help='Whether to produce samples from the model', default=True)
    parser.add_argument('--fast_fid', action='store_true', help='Whether to do fast fid test')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-i', '--image_folder', type=str, default='saved_results', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', default= True, help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--gpu', type = int, default= 0, help="cuda id")
    parser.add_argument('--kappa_id', type=int, default= 1, help="No interaction. Suitable for Slurm Job launcher")
    
    parser.add_argument('--ep_iter', type=int, default= 5, help="EP iterations")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.model_dir)


    args.gpu = 0

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    if args.sample:
        os.makedirs(os.path.join(args.exp, args.image_folder), exist_ok=True)
        quantized_bits = str(new_config.measurements.quantize_bits) + '_bit'
        ep_iter = 'EPiter' +  str(args.ep_iter)
        if not new_config.measurements.quantization:
            quantized_bits = 'linear' 
        args.image_folder = os.path.join(args.exp, args.image_folder, new_config.data.dataset,new_config.measurements.matrix, ep_iter)
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    start_time = time.time()

    print('noise-variance: {}'.format(config.measurements.noise_variance))

    try:
        runner = NCSNRunner(args, config)
        if args.sample:
            runner.sample()
        else:
            raise ValueError('Only sampling is supported!')
    except:
        logging.error(traceback.format_exc())

    return 0
    
    end_time = time.time()
    running_time = end_time - start_time




if __name__ == '__main__':
    sys.exit(main())
