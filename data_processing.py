import os
import dgl
import torch
import pickle
import pysmiles
from collections import defaultdict
import pandas as pd


attribute_names = ['element', 'charge', 'aromatic', 'hcount']

class SmilesDataset(dgl.data.DGLDataset):
    def __init__(self, args, role='reactant', feature_encoder=None, raw_graphs=None):
        self.args = args
        if role=='product' and args.corpus:
            self.mode = args.corpus
        else:
            self.mode = args.mode
        self.feature_encoder = feature_encoder
        self.raw_graphs = raw_graphs
        self.role = role
        self.path = f'dataset/{self.args.dataset}/cache/{self.mode}_{role}_graphs.bin'
        self.graphs = []
        super().__init__(name='Smiles_' + self.mode)

    def to_gpu(self):
        if torch.cuda.is_available():
            print('moving ' + self.mode + ' data to GPU')
            self.graphs = [graph.to('cuda:' + str(self.args.device)) for graph in self.graphs]

    def save(self):
        print(f'saving {self.mode} {self.role} graphs to {self.path}')
        dgl.save_graphs(self.path, self.graphs)

    def load(self):
        print(f'loading {self.mode} {self.role} graphs from {self.path}')
        self.graphs = dgl.load_graphs(self.path)[0]
        self.to_gpu()

    def process(self):
        print('transforming ' + self.mode + ' data from networkx graphs to DGL graphs')
        for i, graph in enumerate(self.raw_graphs):
            if i % 10000 == 0:
                print('%dk' % (i // 1000))
            # transform networkx graphs to dgl graphs
            try:
                graph = networkx_to_dgl(graph, self.feature_encoder)
                self.graphs.append(graph)
            except:
                self.graphs.append(self.graphs[-1])
        self.to_gpu()

    def has_cache(self):
        return os.path.exists(self.path)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


def networkx_to_dgl(raw_graph, feature_encoder):
    # add edges
    src = [s for (s, _) in raw_graph.edges]
    dst = [t for (_, t) in raw_graph.edges]
    graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
    # add node features
    node_features = []
    for i in range(len(raw_graph.nodes)):
        raw_feature = raw_graph.nodes[i]
        numerical_feature = []
        for j in attribute_names:
            if raw_feature[j] in feature_encoder[j]:
                numerical_feature.append(feature_encoder[j][raw_feature[j]])
            else:
                numerical_feature.append(feature_encoder[j]['unknown'])
        node_features.append(numerical_feature)
    node_features = torch.tensor(node_features)
    graph.ndata['feature'] = node_features
    # transform to bi-directed graph with self-loops
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    return graph


def read_data(dataset, mode, featurize=False):
    path = f'dataset/{dataset}/raw/{mode}.csv'
    print('preprocessing %s data from %s' % (mode, path))

    # saving all possible values of each attribute (only for training data)
    all_values = defaultdict(set)
    graphs = []

    df = pd.read_csv(path)
    product_smiles_list = df['product_smiles'].tolist()
    reactant_smiles_list = df['reactant_smiles'].tolist()
    for idx, (product_smiles, reactant_smiles) in enumerate(zip(product_smiles_list, reactant_smiles_list)):

        if int(idx) % 10000 == 0:
            print('%dk' % (int(idx) // 1000))

        # pysmiles.read_smiles() will raise a ValueError: "The atom [se] is malformatted" on USPTO-479k dataset.
        # This is because "Se" is in a aromatic ring, so in USPTO-479k, "Se" is transformed to "se" to satisfy
        # SMILES rules. But pysmiles does not treat "se" as a valid atom and raise a ValueError. To handle this
        # case, I transform all "se" to "Se" in USPTO-479k.
        if '[se]' in reactant_smiles:
            reactant_smiles = reactant_smiles.replace('[se]', '[Se]')
        if '[se]' in product_smiles:
            product_smiles = product_smiles.replace('[se]', '[Se]')

        # use pysmiles.read_smiles() to parse SMILES and get graph objects (in networkx format)
        reactant_graph = pysmiles.read_smiles(reactant_smiles, zero_order_bonds=False)
        product_graph = pysmiles.read_smiles(product_smiles, zero_order_bonds=False)

        if mode == 'train' or featurize:
            # store all values
            for graph in [reactant_graph, product_graph]:
                for attr in attribute_names:
                    for _, value in graph.nodes(data=attr):
                        all_values[attr].add(value)

        graphs.append([reactant_graph, product_graph])

    if mode == 'train' or featurize:
        return all_values, graphs
    else:
        return graphs

def read_graphs(dataset, mode, role):
    path = f'dataset/{dataset}/raw/{mode}.csv'
    print('preprocessing %s data from %s' % (mode, path))

    # saving all possible values of each attribute (only for training data)
    all_values = defaultdict(set)
    graphs = []

    df = pd.read_csv(path)
    smiles_list = df[f'{role}_smiles'].tolist()
    for idx, smiles in enumerate(smiles_list):

        if int(idx) % 10000 == 0:
            print('%dk' % (int(idx) // 1000))

        # pysmiles.read_smiles() will raise a ValueError: "The atom [se] is malformatted" on USPTO-479k dataset.
        # This is because "Se" is in a aromatic ring, so in USPTO-479k, "Se" is transformed to "se" to satisfy
        # SMILES rules. But pysmiles does not treat "se" as a valid atom and raise a ValueError. To handle this
        # case, I transform all "se" to "Se" in USPTO-479k.
        if '[se]' in smiles:
            smiles = smiles.replace('[se]', '[Se]')

        # use pysmiles.read_smiles() to parse SMILES and get graph objects (in networkx format)
        graph = pysmiles.read_smiles(smiles, zero_order_bonds=False)
        graphs.append(graph)

    return graphs

def get_feature_encoder(all_values):
    feature_encoder = {}
    idx = 0
    # key: attribute; values: all possible values of the attribute
    for key, values in all_values.items():
        feature_encoder[key] = {}
        for value in values:
            feature_encoder[key][value] = idx
            idx += 1
        # for each attribute, we add an "unknown" key to handle unknown values during inference
        feature_encoder[key]['unknown'] = idx
        idx += 1

    return feature_encoder


def preprocess(dataset):
    print('preprocessing %s dataset' % dataset)

    # read all data and get all values for attributes
    all_values, train_graphs = read_data(dataset, 'train')
    valid_graphs = read_data(dataset, 'valid')
    test_graphs = read_data(dataset, 'test')

    # get one-hot encoder for attribute values
    feature_encoder = get_feature_encoder(all_values)

    # save feature encoder to disk
    path = f'dataset/{args.dataset}/cache/'
    if not os.path.exists(path):
        os.mkdir(path)
    path = 'dataset/' + dataset + '/cache/feature_encoder.pkl'
    print('saving feature encoder to %s' % path)
    with open(path, 'wb') as f:
        pickle.dump(feature_encoder, f)

    return feature_encoder, train_graphs, valid_graphs, test_graphs

def preprocess_subset(dataset, mode):
    print('preprocessing %s dataset' % dataset)
    path = f'dataset/{dataset}/cache/feature_encoder.pkl'
    with open(path, 'rb') as f:
        feature_encoder = pickle.load(f)

    all_values, graphs = read_data(dataset, mode, True)
    # feature_encoder = get_feature_encoder(all_values)
    
    # path = f'dataset/{dataset}/cache/feature_encoder.pkl'
    # print('saving feature encoder to %s' % path)
    # with open(path, 'wb') as f:
    #     pickle.dump(feature_encoder, f)

    return feature_encoder, graphs


def load_data(args):
    path = 'dataset/' + args.dataset + '/cache/feature_encoder.pkl'
    print('loading feature encoder from %s' % path)
    with open(path, 'rb') as f:
        feature_encoder = pickle.load(f)

    # preprocess reactant graphs
    reactant_path = f'dataset/{args.dataset}/cache/{args.mode}_reactant_graphs.bin'
    if os.path.exists(reactant_path):
        reactant_dataset = SmilesDataset(args)
    else:
        path = f'dataset/{args.dataset}/cache/'
        if not os.path.exists(path):
            os.mkdir(path)
        graphs = read_graphs(args.dataset, args.mode, 'reactant')
        reactant_dataset = SmilesDataset(args, 'reactant', feature_encoder, graphs)

    # preprocess product graphs
    if args.corpus:
        product_path = f'dataset/{args.dataset}/cache/{args.corpus}_product_graphs.bin'
    else:
        product_path = f'dataset/{args.dataset}/cache/{args.mode}_product_graphs.bin'
    if os.path.exists(product_path):
        product_dataset = SmilesDataset(args, 'product')
    else:
        path = f'dataset/{args.dataset}/cache/'
        if not os.path.exists(path):
            os.mkdir(path)
        graphs = read_graphs(args.dataset, args.corpus, 'product')
        product_dataset = SmilesDataset(args, 'product', feature_encoder, graphs)

    return feature_encoder, (reactant_dataset, product_dataset)
    
def load_data_with_smiles(args):
    data = load_data(args)

    df = pd.read_csv(f'dataset/{args.dataset}/raw/{args.mode}.csv')
    smiles = [df['reactant_smiles'], df['product_smiles']]

    return data, smiles

def load_identifiers(args):
    df = pd.read_csv(f'dataset/{args.dataset}/raw/{args.mode}.csv')
    reactant_identifiers = {
        'smiles': df['reactant_smiles'].tolist(),
        'iupac':df['reactant_iupac'].tolist()
    }
    product_identifiers = {
        'smiles': df['product_smiles'].tolist(),
        'iupac':df['product_iupac'].tolist()
    }

    if args.corpus:
        df = pd.read_csv(f'dataset/{args.dataset}/raw/{args.corpus}.csv')
    corpus_identifiers = {
        'smiles': df['product_smiles'].tolist(),
        'iupac':df['product_iupac'].tolist()
    }

    identifiers = {
        'reactant': reactant_identifiers,
        'product': product_identifiers,
        'corpus': corpus_identifiers
    }
    return identifiers