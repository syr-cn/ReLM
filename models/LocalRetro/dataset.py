import os
import pandas as pd

from rdkit import Chem

import torch
import sklearn
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs
from dgllife.utils import smiles_to_bigraph
from functools import partial

def canonicalize_rxn(rxn):
    canonicalized_smiles = []
    r, p = rxn.split('>>')
    for s in [r, p]:
        mol = Chem.MolFromSmiles(s)
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
        canonicalized_smiles.append(Chem.MolToSmiles(mol))
    return '>>'.join(canonicalized_smiles)

class USPTOTestDataset(object):
    def __init__(self, args, mode, role, node_featurizer, edge_featurizer, load=True, log_every=1000, smiles_to_graph=None):
        df = pd.read_csv(f'dataset/{args.dataset}/raw/{mode}.csv')
        self.smiles = df[f'{role}_smiles']

        cache_file_path = f'dataset/{args.dataset}/cache_lr/'
        if not os.path.exists(cache_file_path):
            os.mkdir(cache_file_path)
        self.cache_file_path = f'dataset/{args.dataset}/cache_lr/{mode}_{role}_graphs.bin'

        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every):
        if os.path.exists(self.cache_file_path) and load:
            print('Loading previously saved test dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
        else:
            print('Processing test dgl graphs from scratch...')
            self.graphs = []
            for i, s in enumerate(self.smiles):
                if (i + 1) % log_every == 0:
                    print('Processing molecule %d/%d' % (i+1, len(self.smiles)))
                self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer,
                                                   edge_featurizer=edge_featurizer, canonical_atom_order=False))
            save_graphs(self.cache_file_path, self.graphs)

    def __getitem__(self, item):
            return self.graphs[item]

    def __len__(self):
            return len(self.smiles)