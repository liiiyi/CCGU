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

    Legacy defaults are preserved; the scripts under ``reproduction/scripts/`` pass
    experiment-specific values explicitly rather than changing them.
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
    parser.add_argument('--random_seed', type=int, default=20221012,
                        help='seed for every RNG the pipeline uses (numpy, random, torch, '
                             'community detection, the train/val/test split and the sampled '
                             'unlearning request).  The default is the value main.py used to '
                             'hard-code; the reproduction runs use 4 and 5.')
    parser.add_argument('--log_file', type=str, default=None,
                        help='also append the run log to this file')
    parser.add_argument('--metrics_file', type=str, default=None,
                        help='write the metrics of this stage to this JSON path')

    ########################## unlearning task parameters ######################
    parser.add_argument('--unlearn_task', type=str, default='node', choices=["node"],
                        help='CGE currently implements node unlearning')
    parser.add_argument('--unlearn_ratio', type=float, default=0.1,
                        help='fraction of the eligible training nodes when in (0, 1], absolute '
                             'node count when > 1.  The paper unlearns 0.5%% per batch, i.e. '
                             '--unlearn_ratio 0.005.')

    ########################### nucleus graph paramters ############################
    parser.add_argument('--agg_feat', type=str, default='pca', choices=["pca", "mean"])
    parser.add_argument('--agg_label', type=str, default='th', choices=["th", "all", "km"])
    parser.add_argument('--agg_edge', type=str, default='rubost', choices=["rubost", "jaccard", "opt"])
    parser.add_argument('--partition', type=str, default='oslom',
                        choices=["slpa", "oslom", "nikm", "lpa", "louvain", "test", "infomap",
                                 "ccp", "gae"],
                        help="'ccp' is the maintained implementation of the coarse-to-fine "
                             "overlapping community process described in the paper; 'gae' is a "
                             "post-paper graph-auto-encoder detector; 'oslom' (the default), "
                             "'slpa' and 'lpa' are label-propagation implementations retained "
                             "with their original behaviour.  See exp/methods/README.md.")
    parser.add_argument('--th_sim2edge', type=float, default=-1, help='do not use threshold when < 0, automatically when 0')
    parser.add_argument('--test_edge_method', type=int, default=2, choices=[0, 1, 2, 3],
                        help='0 reproduces Equation (11) exactly (sqrt-degree normalisation); 1 and 2 '
                             'are log-normalised variants; 3 drops the robustness term.  With the '
                             'default --th_sim2edge -1 and --use_edge_weight False this choice does '
                             'not change the mapped-graph topology, only the stored scores.')
    parser.add_argument('--use_edge_weight', type=str2bool, default=False)
    parser.add_argument('--use_feat_rb', type=str2bool, default=False)

    ############ community-detection parameters for --partition ccp ################
    parser.add_argument('--ccp_resolution', type=float, default=1.0,
                        help='resolution of the coarse Louvain stage')
    parser.add_argument('--ccp_fine_resolution', type=float, default=1.0,
                        help='resolution of the fine-grained refinement stage')
    parser.add_argument('--ccp_theta', type=int, default=20,
                        help='communities larger than this are refined further')
    parser.add_argument('--ccp_max_communities_per_node', type=int, default=3,
                        help='maximum number of communities a node may belong to')
    parser.add_argument('--ccp_overlap_threshold', type=float, default=0.35,
                        help='minimum belonging coefficient for an extra membership')
    parser.add_argument('--ccp_propagation_steps', type=int, default=2,
                        help='graph-smoothing steps for the attribute belonging term')
    parser.add_argument('--ccp_max_community_size', type=int, default=0,
                        help='post-paper engineering extension: hard cap on community size, '
                             'enforced after the overlap stage by deterministic BFS chunking. '
                             '0 (default) disables it and reproduces the behaviour without it. '
                             'Must be >= --ccp_theta.')
    parser.add_argument('--ccp_tail_min_size', type=int, default=0,
                        help='post-paper engineering extension: opt-in tail repair.  Communities '
                             'smaller than this are merged into the neighbouring community with '
                             'the highest label-free attachment score (structural overlap x '
                             'propagated-feature centroid similarity).  Protected evaluation '
                             'singletons are never merged.  0 (default) disables it.')
    parser.add_argument('--ccp_protect_eval_nodes', type=str, default='none',
                        choices=['none', 'test', 'test_val'],
                        help="keep the original graph's held-out nodes in singleton "
                             "communities so that exp_train's test_communities branch can "
                             "evaluate the mapped model on real held-out original nodes.  "
                             "'none' (default) keeps the historical behaviour of "
                             "aggregating every node.")
    parser.add_argument('--ccp_conductance_threshold', type=float, default=-1.0,
                        help="refine only communities whose conductance exceeds this value; "
                             "negative refines every oversized community (the paper's "
                             "small-dataset variant)")

    parser.add_argument('--louvain_resolution', type=float, default=2.0,
                        help="resolution for --partition louvain; kept separate from "
                             "--ccp_resolution so the legacy wrapper's 2.0 is preserved")

    ############ graph-auto-encoder detector for --partition gae ###################
    # Post-paper extension.  The paper uses a GAE only for its Information Retention
    # metric, never for community detection.
    parser.add_argument('--gae_hidden_dim', type=int, default=64,
                        help='hidden width of the GAE encoder')
    parser.add_argument('--gae_latent_dim', type=int, default=16,
                        help='latent width the clustering runs on')
    parser.add_argument('--gae_epochs', type=int, default=100,
                        help='link-reconstruction training epochs')
    parser.add_argument('--gae_learning_rate', type=float, default=0.01)
    parser.add_argument('--gae_round_decimals', type=int, default=6,
                        help='decimals the L2-normalised embedding is rounded to before '
                             'clustering, so that ~1e-07 non-associative reduction wobble '
                             'in DGL message passing cannot move a k-means boundary')
    parser.add_argument('--gae_device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help="'auto' uses CUDA when torch reports it available and "
                             "falls back to CPU; k-means and the overlap stage are CPU "
                             "either way")

    ###################### partition quality gate ##################################
    parser.add_argument('--partition_min_communities', type=int, default=8,
                        help='fail-fast floor on the number of detected communities')
    parser.add_argument('--partition_max_retries', type=int, default=0,
                        help='if > 0, re-run a partition that fails the quality gate with seed+1, '
                             'up to this many extra attempts.  Every attempt is logged and the '
                             'gate only ever looks at the partition, never at model quality.')

    ########################## training parameters ###########################
    parser.add_argument('--is_split', type=str2bool, default=True, help='splitting train/test data')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--use_test_neighbors', type=str2bool, default=True)
    parser.add_argument('--is_train_target_model', type=str2bool, default=True)
    parser.add_argument('--is_retrain', type=str2bool, default=True)
    parser.add_argument('--is_use_node_feature', type=str2bool, default=False)
    parser.add_argument('--is_use_batch', type=str2bool, default=True, help="Use batch train GNN models.")
    parser.add_argument('--target_model', type=str, default='GCN', choices=["SAGE", "GAT", 'MLP', "GCN", "GIN", "SGC"])
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_weight_decay', type=float, default=0,
                        help="the paper's appendix reports 0.001; the reproduction scripts pass it "
                             "explicitly rather than changing this default")
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_runs', type=int, default=1,
                        help='not consumed by a single invocation; the reproduction driver repeats '
                             'the pipeline over --random_seed instead')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=64)

    ########################## GIF parameters ###########################
    parser.add_argument('--iteration', type=int, default=5)
    parser.add_argument('--scale', type=int, default=50)
    parser.add_argument('--damp', type=float, default=0.0)

    args = vars(parser.parse_args())

    return args
