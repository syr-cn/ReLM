import os
import torch
import pickle
import data_processing
import numpy as np
from model import load_model
from copy import deepcopy
from dgl.dataloading import GraphDataLoader



def evaluate_gnn(args, data):
    feature_encoder, valid_data, test_data = data

    model = load_model(args)

    model.eval()
    evaluate(model, 'valid', valid_data, args)
    evaluate(model, 'test', test_data, args)


def evaluate(model, mode, data, args):
    model.eval()
    with torch.no_grad():
        # calculate embeddings of all products as the candidate pool
        all_product_embeddings = []
        product_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        for _, product_graphs in product_dataloader:
            product_embeddings = model(product_graphs)
            all_product_embeddings.append(product_embeddings)
        all_product_embeddings = torch.cat(all_product_embeddings, dim=0)
        # rank
        all_rankings = []
        reactant_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        i = 0
        for reactant_graphs, _ in reactant_dataloader:
            reactant_embeddings = model(reactant_graphs)
            ground_truth = torch.unsqueeze(torch.arange(i, min(i + args.batch_size, len(data))), dim=1)
            i += args.batch_size
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda(args.device)
            dist = torch.cdist(reactant_embeddings, all_product_embeddings, p=2)
            sorted_indices = torch.argsort(dist, dim=1)
            rankings = ((sorted_indices == ground_truth).nonzero()[:, 1] + 1).tolist()
            all_rankings.extend(rankings)

        # calculate metrics
        all_rankings = np.array(all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        mr = float(np.mean(all_rankings))
        h1 = float(np.mean(all_rankings <= 1))
        h3 = float(np.mean(all_rankings <= 3))
        h5 = float(np.mean(all_rankings <= 5))
        h10 = float(np.mean(all_rankings <= 10))

        print('%s  mrr: %.4f  mr: %.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f' % (mode, mrr, mr, h1, h3, h5, h10))
        return mrr
