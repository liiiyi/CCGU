import torch
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
from tqdm import tqdm
import logging
from lib_dataset.data_store import DataStore
import config
import os


class GetEmbed:
    def __init__(self, graph, args):
        self.logger = logging.getLogger('GetEmbed')
        self.data_store = DataStore(args)
        self.graph = graph
        self.args = args
        self.device = torch.device(f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu')
        self.graph = self.graph.to(self.device)
        self.embeddings_file = config.PROCESSED_DATA_PATH + self.args['dataset_name'] + "_embeddings"

    def get_model(self):
        num_feats = self.graph.ndata['feat'].shape[1]
        num_hidden = 128
        if self.args['dataset_name'] in {'reddit', 'facebook'}:
            model = SAGEConv(num_feats, num_hidden, aggregator_type='mean').to(self.device)
        elif self.args['dataset_name'] in {'cora', 'citeseer'}:
            model = GraphConv(num_feats, num_hidden).to(self.device)
        elif self.args['dataset_name'] in {'pubmed', 'CS'}:
            model = GATConv(num_feats, num_hidden, num_heads=1).to(self.device)
        else:
            raise ValueError(f"Unsupported GNN model")
        return model

    def generate_embeddings(self):
        if os.path.exists(self.embeddings_file):
            embeddings = self.data_store.load_embeddings()
        else:
            self.logger.info('Generating embeddings...')
            model = self.get_model()
            features = self.graph.ndata['feat'].to(self.device)
            with torch.no_grad():
                if isinstance(model, GATConv):
                    embeddings = model(self.graph, features).mean(dim=1)
                else:
                    embeddings = model(self.graph, features)
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
            self.data_store.save_embeddings(embeddings)
        return embeddings
