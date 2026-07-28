import logging
import os
import random
import sys

import numpy as np
import torch

from parameter_parser import parameter_parser


def _set_random_seed(seed):
    """Seed every RNG the pipeline touches and pin cuDNN to deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("set random seed %d" % seed)


def config_logger(save_name, log_file=None):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')

    # create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        directory = os.path.dirname(os.path.abspath(log_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logging.info("run: %s", save_name)


def _select_device(requested):
    """Fail early with a readable message instead of deep inside a CUDA call."""
    if not torch.cuda.is_available():
        logging.info("CUDA is not available, running on CPU")
        return
    available = torch.cuda.device_count()
    if requested < 0 or requested >= available:
        raise SystemExit(
            "--cuda {} is out of range: this host exposes {} CUDA device(s), so valid "
            "values are 0..{}".format(requested, available, available - 1)
        )
    torch.cuda.set_device(requested)


def _run_removed_stage(name, module_path):
    raise SystemExit(
        "--exp {} needs {}, which is not part of this public release (see README -> "
        "Running the pipeline).  The supported stages are Partition, Train and "
        "Unlearn.".format(
            name, module_path
        )
    )


if __name__ == "__main__":
    args = parameter_parser()

    _set_random_seed(args["random_seed"])

    # config the logger
    logger_name = "_".join((args['dataset_name'], args['partition'], args['target_model'],
                            args['unlearn_task'], str(args['unlearn_ratio']),
                            'seed' + str(args['random_seed'])))
    config_logger(logger_name, log_file=args["log_file"])

    torch.set_num_threads(args["num_threads"])
    _select_device(args["cuda"])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if args["exp"].lower() == "partition":
        from exp.exp_partition import GraphCommunityPartition
        GraphCommunityPartition(args)
    elif args["exp"].lower() == "train":
        from exp.exp_train import TrainModel
        TrainModel(args)
    elif args["exp"].lower() == "draw":
        _run_removed_stage("Draw", "exp/exp_draw.py")
    elif args["exp"].lower() == "eva":
        _run_removed_stage("eva", "exp/Evaluate.py")
    elif args["exp"].lower() == "unlearn":
        from exp.exp_unlearn import Unlearn
        Unlearn(args)
    else:
        raise SystemExit(
            "unknown --exp {!r}; choose Partition, Train or Unlearn".format(args["exp"])
        )
