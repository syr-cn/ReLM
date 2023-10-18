import os
import torch
import pickle
# import data_processing
import numpy as np
from model import load_model
from copy import deepcopy
from torch_geometric.data import DataLoader
from loader import ReactionDataset
import data_processing


# torch.set_printoptions(profile="full", linewidth=100000, sci_mode=False)

def train(args):
    data = data_processing.preprocess(args.dataset)
    feature_encoder, train_data, valid_data, test_data = data
    feature_len = sum([len(feature_encoder[key]) for key in data_processing.attribute_names])
    print(feature_len)
    assert 0
    
    

def train_pyG(args):
    dataset1 = ReactionDataset(root=f'dataset/{args.dataset}/', role='reactant', mode='train')
    dataset2 = ReactionDataset(root=f'dataset/{args.dataset}/', role='prod', mode='train')
    loader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=False)
    
    dataset_val = (
        ReactionDataset(root=f'dataset/{args.dataset}/', role='reactant', mode='valid'),
        ReactionDataset(root=f'dataset/{args.dataset}/', role='prod', mode='valid')
    )
    
    dataset_test = (
        ReactionDataset(root=f'dataset/{args.dataset}/', role='reactant', mode='test'),
        ReactionDataset(root=f'dataset/{args.dataset}/', role='prod', mode='test')
    )

    model = load_model(args, load=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if torch.cuda.is_available():
        model = model.to(args.device)

    best_model_params = None
    best_val_mrr = 0
    print('start training\n')

    print('initial case:')
    # model.eval()
    # evaluate(model, 'valid', valid_data, args)
    # evaluate(model, 'test', test_data, args)
    print()
    for i in range(args.epoch):
        print('epoch %d:' % i, flush=True)

        # train
        model.train()
        for reactant_graphs, product_graphs in zip(loader1, loader2):
            reactant_graphs = reactant_graphs.to(args.device)
            product_graphs = product_graphs.to(args.device)
            reactant_embeddings = model(reactant_graphs)
            product_embeddings = model(product_graphs)
            loss = calculate_loss(reactant_embeddings, product_embeddings, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate on the validation set
        val_mrr = evaluate(model, 'valid', dataset_val, args)
        test_mrr = evaluate(model, 'test', dataset_test, args)

        # save the best model
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_model_params = deepcopy(model.state_dict())

        print()

    # evaluation on the test set
    print('final results on the test set:')
    model.load_state_dict(best_model_params)
    evaluate(model, 'test', dataset_test, args)
    print()

    # save the model, hyperparameters, and feature encoder to disk
    torch.save(model.state_dict(), args.config['model_file'])


def calculate_loss(reactant_embeddings, product_embeddings, args):
    n = reactant_embeddings.shape[0]
    dist = torch.cdist(reactant_embeddings, product_embeddings, p=2)
    pos = torch.diag(dist)
    mask = torch.eye(n)
    if torch.cuda.is_available():
        mask = mask.to(args.device)
    neg = (1 - mask) * dist + mask * args.margin
    neg = torch.relu(args.margin - neg)
    loss = torch.mean(pos) + torch.sum(neg) / n / (n - 1)
    return loss


def evaluate(model, mode, data, args):
    reactant_dataloader = DataLoader(data[0], batch_size=args.batch_size, shuffle=False)
    product_dataloader = DataLoader(data[1], batch_size=args.batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        # calculate embeddings of all products as the candidate pool
        all_product_embeddings = []
        for product_graphs in product_dataloader:
            product_embeddings = model(product_graphs.to(args.device))
            all_product_embeddings.append(product_embeddings)
        all_product_embeddings = torch.cat(all_product_embeddings, dim=0)
        # rank
        all_rankings = []
        i = 0
        for reactant_graphs in reactant_dataloader:
            reactant_embeddings = model(reactant_graphs.to(args.device))
            ground_truth = torch.unsqueeze(torch.arange(i, min(i + args.batch_size, len(data[0]))), dim=1)
            i += args.batch_size
            if torch.cuda.is_available():
                ground_truth = ground_truth.to(args.device)
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