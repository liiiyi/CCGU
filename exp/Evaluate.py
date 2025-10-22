from exp.exp import Exp
import logging
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import dgl
from exp.methods.Evaluator import CommunityEvaluator as CE
import pickle
import config
from exp.Eva.E_Graph import GraphRepresentativenessEvaluator as CRE

class Evaluate(Exp):
    def __init__(self, args):
        super(Evaluate, self).__init__(args)
        self.logger = logging.getLogger('Evaluate')
        self.args = args
        self.load_data()
        self.eva_graph()


    def load_data(self):
        load_communities_data = self.data_store.load_communities_info()
        self.c2n = load_communities_data['c2n']
        self.graph = self.data_store.load_graph_from_dgl()
        address = config.COM_PATH + 'community_' + self.args['dataset_name']
        # self.c2n = pickle.load(open(address, 'rb'))
        # self.nucleus_graph = self.data_store.load_nucleus_graph()

    def eva_graph(self):
        test = 'cge'
        if test is 'cge':
            cre = CRE(self.graph, subgraph=self.nucleus_graph)
            score = cre.evaluate_all()
            print(score)
        else:
            score = 0
            crore = 0
            max_s = 0
            max_c = 0
            for i in range(len(self.c2n)):
                c_nodes = self.c2n[i]
                cre = CRE(self.graph, community_nodes=c_nodes)
                s, c = cre.evaluate_all()["representativeness_score"], cre.evaluate_all()["information_retention"]
                score += s
                crore += c
                print(f'c{i}:{s}, {c}')
                if s > max_s:
                    max_s = s
                    print(f'update max score:{max_s}')
                if c > max_c:
                    max_c = c
                    print(f'update max score:{max_c}')
                print(f'mean_c:{crore/(i+1)}')
            print(f'final:{score/len(self.c2n), crore/len(self.c2n)}, max:{max_s, max_c}')


