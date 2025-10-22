import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyper-parameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--is_vary', type=bool, default=False, help='control whether to use multiprocess')
    parser.add_argument('--cuda', type=int, default=1, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--exp', type=str, default='Draw', choices=["Partition", "Train", "Draw", "eva", "Unlearn"])
    parser.add_argument('--dgl_data', type=str2bool, default=True, help='load dataset from DGL')
    parser.add_argument('--dataset_name', type=str, default='reddit',
                        choices=["cora", "citeseer", "pubmed", "cs", "aifb", "reddit"])
    parser.add_argument('--if_embed', type=bool, default=False)

    ########################## unlearning task parameters ######################
    # parser.add_argument('--unlearn_task', type=str, default='edge', choices=["edge", "node", 'feature'])
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)


    ########################### nucleus graph paramters ############################
    parser.add_argument('--agg_feat', type=str, default='pca', choices=["pca", "mean"])
    parser.add_argument('--agg_label', type=str, default='th', choices=["th", "all", "km"])
    parser.add_argument('--agg_edge', type=str, default='rubost', choices=["rubost", "jaccard", "opt"])
    parser.add_argument('--partition', type=str, default='oslom', choices=["slpa", "oslom", "nikm", "lpa", "louvain", "test", "infomap"])
    parser.add_argument('--th_sim2edge', type=float, default=-1, help='do not use threshold when < 0, automatically when 0')
    parser.add_argument('--test_edge_method', type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument('--use_edge_weight', type=str2bool, default=False)
    parser.add_argument('--use_feat_rb', type=str2bool, default=False)

    ########################## training parameters ###########################
    parser.add_argument('--is_split', type=str2bool, default=True, help='splitting train/test data')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--use_test_neighbors', type=str2bool, default=True)
    parser.add_argument('--is_train_target_model', type=str2bool, default=True)
    parser.add_argument('--is_retrain', type=str2bool, default=True)
    parser.add_argument('--is_use_node_feature', type=str2bool, default=False)
    parser.add_argument('--is_use_batch', type=str2bool, default=True, help="Use batch train GNN models.")
    parser.add_argument('--target_model', type=str, default='GCN', choices=["SAGE", "GAT", 'MLP', "GCN", "GIN","SGC"])
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_weight_decay', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=64)

    ########################## GIF parameters ###########################
    parser.add_argument('--iteration', type=int, default=5)
    parser.add_argument('--scale', type=int, default=50)
    parser.add_argument('--damp', type=float, default=0.0)

    args = vars(parser.parse_args())

    return args
