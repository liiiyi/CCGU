import logging
import os
import torch
import sys
import numpy as np
import random

from parameter_parser import parameter_parser
from exp.exp_partition import GraphCommunityPartition
from exp.exp_train import TrainModel
from exp.exp_draw import ExpDraw
from exp.Evaluate import Evaluate
# from exp.exp_unlearn import Unlearn

def _set_random_seed(seed=2022):
    
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed")

def config_logger(save_name):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')

    # create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
   

if __name__ == "__main__":
    args = parameter_parser()

    _set_random_seed(20221012)

    # config the logger
    logger_name = "_".join((args['dataset_name'], str(args['test_ratio']), args['target_model'], args['unlearn_task'], str(args['unlearn_ratio'])))
    config_logger(logger_name)
    logging.info(logger_name)

    torch.set_num_threads(args["num_threads"])
    torch.cuda.set_device(args["cuda"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


    if args["exp"].lower() == "partition":
        GraphCommunityPartition(args)
    elif args["exp"].lower() == "train":
        TrainModel(args)
    elif args["exp"].lower() == "draw":
        ExpDraw(args)
    elif args["exp"].lower() == "eva":
        Evaluate(args)
    '''
    elif args["exp"].lower() == "unlearn":
        Unlearn(args)
    '''
