import os
import torch
import random
import data_processing
import numpy as np
from model import load_model, LLM
from copy import deepcopy
from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader
from prompt_generator import *
from utils import *
from models.LocalRetro.utils import load_dataset as load_localretro_dataset
# from models.megan.dataset import load_megan_dataset
from tqdm import tqdm
import pandas as pd
import time

from torch_geometric.data import DataLoader
from loader import ReactionDataset
from collections import defaultdict
import json

subset_len = 500
subset_len = -1

def evaluate_llm(args):
    if args.model == 'MolR_pyG':
        evaluate_llm_pyG(args)
        return
    elif args.model == 'LocalRetro':
        evaluate_llm_LocalRetro(args)
        return
    elif args.model == 'Megan':
        evaluate_llm_Megan(args)
        return
    data = data_processing.load_data(args)[1]
    identifiers = data_processing.load_identifiers(args)
    
    context_args = deepcopy(args)
    if args.context:
        context_args.mode = args.context
        context_args.corpus = args.context
    else:
        context_args.corpus = args.mode
    context_data = data_processing.load_data(context_args)[1]
    context_identifiers = data_processing.load_identifiers(context_args)

    model = load_model(args)

    model.eval()
    # evaluate(model, 'test', test_data, test_identifiers, args)
    evaluate(model, args.mode, data, identifiers, context_data, context_identifiers, args)


def evaluate_llm_LocalRetro(args):
    data = load_localretro_dataset(args)
    identifiers = data_processing.load_identifiers(args)
    
    context_args = deepcopy(args)
    if args.context:
        context_args.mode = args.context
        context_args.corpus = args.context
    else:
        context_args.corpus = args.mode
    context_data = load_localretro_dataset(context_args)
    context_identifiers = data_processing.load_identifiers(context_args)

    model = load_model(args)
    model.eval()
    evaluate(model, args.mode, data, identifiers, context_data, context_identifiers, args)

def evaluate_llm_Megan(args):
    pass
    # data = load_megan_dataset(args)
    # identifiers = data_processing.load_identifiers(args)
    
    # context_args = deepcopy(args)
    # if args.context:
    #     context_args.mode = args.context
    #     context_args.corpus = args.context
    # else:
    #     context_args.corpus = args.mode
    # context_data = load_megan_dataset(context_args)
    # context_identifiers = data_processing.load_identifiers(context_args)

    # model = load_model(args)
    # model.eval()
    # evaluate(model, args.mode, data, identifiers, context_data, context_identifiers, args)

def evaluate_llm_pyG(args):
    data = (
        ReactionDataset(root=f'dataset/{args.dataset}/', role='reactant', mode=args.mode),
        ReactionDataset(root=f'dataset/{args.dataset}/', role='prod', mode=args.mode)
    )
    identifiers = data_processing.load_identifiers(args)
    model = load_model(args)

    model.eval()
    # evaluate(model, 'test', test_data, test_identifiers, args)
    # evaluate_pyG(model, args.mode, data, identifiers, args)

def simplify_product(x, identifiers, args=None):
    products_smiles = identifiers['product']['smiles']
    corpus_smiles = identifiers['corpus']['smiles']
    corpus_iupac = identifiers['corpus']['iupac']
    assert x.shape[0] == len(corpus_smiles), f'{x.shape[0]=}, which does not match {len(corpus_smiles)=}'

    label, y = [], []
    table = defaultdict(lambda: None)
    smiles_corpus = []
    iupac_corpus = []
    for idx, (smiles, iupac) in enumerate(zip(corpus_smiles, corpus_iupac)):
        if table[smiles] is None:
            table[smiles] = len(smiles_corpus)
            smiles_corpus.append(smiles)
            iupac_corpus.append(iupac)
            y.append(x[idx].reshape(1, -1))
    identifiers['corpus']['smiles'] = smiles_corpus
    identifiers['corpus']['iupac'] = iupac_corpus

    fail_count = 0
    for product in products_smiles:
        if table[product]:
            label.append(table[product])
        else:
            label.append(0)
            fail_count += 1
    if fail_count/len(label)>0.1:
        print(f'warning! {float(fail_count/len(label)*100):.2}% reactions can\'t find their product in the corpus.')

    y = torch.cat(y).to(x.device)
    label = torch.tensor(label, dtype=torch.int32, device=x.device)
    return y, label

@torch.no_grad()
def evaluate(model, mode, data, identifiers, context_data, context_identifiers, args):
    model.eval()

    # Calculation for in-context dataset (maybe training set)
    reactant_data, product_data = context_data
    print(f'\nnumber of samples in <{args.context}> reactant dataset (as context examples): {len(reactant_data)}.')
    print(f'number of samples in <{args.context}> product dataset (as context examples): {len(product_data)}.')
    
    context_product_embeddings = []
    product_dataloader = GraphDataLoader(product_data, batch_size=args.batch_size)
    for product_graphs in product_dataloader:
        product_embeddings = model(product_graphs)
        context_product_embeddings.append(product_embeddings)
    context_product_embeddings = torch.cat(context_product_embeddings, dim=0)

    reactant_dataloader = GraphDataLoader(reactant_data, batch_size=args.batch_size)
    i = 0
    context_candidates = []
    context_scores = []
    context_answers = []
    context_reactant_embeddings = []
    for reactant_graphs in reactant_dataloader:
        reactant_embeddings = model(reactant_graphs)
        context_reactant_embeddings.append(reactant_embeddings)
        dist = torch.cdist(reactant_embeddings, context_product_embeddings, p=2)
        dist, sorted_indices = torch.sort(dist, dim=1)
        ground_truth = torch.unsqueeze(torch.arange(i, min(i + args.batch_size, len(reactant_data))), dim=1).cuda(args.device)
        answer = (sorted_indices == ground_truth).nonzero()[:, 1]
        candidates = sorted_indices[:, :args.k]
        candidates[answer>=args.k, 0] = torch.squeeze(ground_truth[answer>=args.k])
        answer[answer>=args.k] = 0

        context_candidates.append(candidates)
        context_scores.append(dist[:, :args.k])
        context_answers.append(answer)
        i += args.batch_size

    context_answers = torch.cat(context_answers, dim=0)
    context_candidates = torch.cat(context_candidates, dim=0)
    context_scores = torch.cat(context_scores, dim=0)
    del context_product_embeddings
    torch.cuda.empty_cache()
    context_reactant_embeddings = torch.cat(context_reactant_embeddings, dim=0)

    # Calculation for test dataset
    reactant_data, product_data = data
    print(f'\nnumber of samples in <{mode}> reactant dataset: {len(reactant_data)}.')
    print(f'number of samples in <{args.corpus}> product corpus: {len(product_data)}.')
    # calculate embeddings of all products as the candidate pool
    all_product_embeddings = []
    product_dataloader = GraphDataLoader(product_data, batch_size=args.batch_size)
    for product_graphs in product_dataloader:
        product_embeddings = model(product_graphs)
        all_product_embeddings.append(product_embeddings)
    all_product_embeddings = torch.cat(all_product_embeddings, dim=0)
    
    product_set_embeddings, labels = simplify_product(all_product_embeddings, identifiers, args)
    print(f'number of unique products in {mode} dataset: {product_set_embeddings.shape[0]}.\n')

    reactant_dataloader = GraphDataLoader(reactant_data, batch_size=args.batch_size)
    i = 0
    candidates = []
    scores = []
    answers = []
    sim = []
    for reactant_graphs in reactant_dataloader:
        reactant_embeddings = model(reactant_graphs)

        dist = torch.cdist(reactant_embeddings, product_set_embeddings, p=2)
        dist, sorted_indices = torch.sort(dist, dim=1)
        ground_truth = torch.unsqueeze(labels[i:min(i + args.batch_size, len(reactant_data))], dim=1)
        answer = (sorted_indices == ground_truth).nonzero()[:, 1]
        answer[answer>=args.k] = args.k

        w1 = reactant_embeddings.norm(p=2, dim=1, keepdim=True)
        w2 = context_reactant_embeddings.norm(p=2, dim=1, keepdim=True)
        context_dist = torch.mm(reactant_embeddings, context_reactant_embeddings.T)/(w1*w2.T)
        _, context_indices = torch.sort(context_dist, dim=1, descending=True)
        if args.context:
            sim.append(context_indices[:, :int(args.num_sim*args.query_num)])
        else:
            sim.append(context_indices[:, 1:int(args.num_sim*args.query_num)+1])
        candidates.append(sorted_indices[:, :args.k])
        scores.append(dist[:, :args.k])
        answers.append(answer)
        i += args.batch_size
    del all_product_embeddings
    del product_set_embeddings
    del context_reactant_embeddings
    torch.cuda.empty_cache()

    answers = torch.cat(answers, dim=0)
    candidates = torch.cat(candidates, dim=0)
    scores = torch.cat(scores, dim=0)
    sim = torch.cat(sim, dim=0)

    global subset_len
    if args.llm =='gpt':
        if subset_len < 0:
            subset_len = 500
    if subset_len<=0:
        subset_len = len(reactant_data)
    if args.context=='random':
        try:
            sim = []
            for idx in range(subset_len, subset_len+50):
                if answers[idx]<args.k:
                    sim.append(idx)
                if len(sim)==int(args.num_sim*args.query_num):
                    break
        except:
            sim = []
            for idx in range(0, len(reactant_data)):
                if answers[idx]<args.k:
                    sim.append(idx)
                if len(sim)==int(args.num_sim*args.query_num):
                    break
        sim = torch.tensor(sim).expand([len(reactant_data), -1])
    
    sim = sim[:subset_len, :]
    hit_k = torch.mean((answers<args.k).float()).item()
    gnn_acc = torch.mean((answers==0).float()).item()
    print(f'{mode} gnn accuracy: {gnn_acc:.5f}')
    print(f'{mode} hit@{args.k}: {hit_k:.5f}')

    # assert 0
    acc = prompt_reasoning(
        sim,
        (candidates, scores, answers, identifiers),
        (context_candidates, context_scores, context_answers, context_identifiers),
        args
    )
    print(f'{mode} gnn+llm accuracy: {acc:.5f}')

'''
def evaluate_pyG(model, mode, data, identifiers, args):
    # deprecated
    print(f'\nnumber of samples in {mode} dataset: {len(data[0])}.')
    model.eval()
    device = args.device
    with torch.no_grad():
        # calculate embeddings of all products as the candidate pool
        all_product_embeddings = []
        # product_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        product_dataloader = DataLoader(data[1], batch_size=args.batch_size)
        # for _, product_graphs in product_dataloader:
        for product_graphs in product_dataloader:
            product_embeddings = model(product_graphs.to(device))
            all_product_embeddings.append(product_embeddings)
        all_product_embeddings = torch.cat(all_product_embeddings, dim=0)
        
        product_set_embeddings, labels = simplify_product(all_product_embeddings, identifiers[1]['smiles'])

        # reactant_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        reactant_dataloader = DataLoader(data[0], batch_size=args.batch_size)
        i = 0
        candidates = []
        scores = []
        answers = []
        # for reactant_graphs, _ in reactant_dataloader:
        for reactant_graphs in reactant_dataloader:
            reactant_embeddings = model(reactant_graphs.to(device))

            dist = torch.cdist(reactant_embeddings, product_set_embeddings, p=2)
            dist, sorted_indices = torch.sort(dist, dim=1)
            ground_truth = torch.unsqueeze(labels[i:min(i + args.batch_size, len(data[0]))], dim=1)
            answer = (sorted_indices == ground_truth).nonzero()[:, 1]
            answer[answer>=args.k] = args.k

            candidates.append(sorted_indices[:, :args.k])
            scores.append(dist[:, :args.k])
            answers.append(answer)
            i += args.batch_size
    
        # TODO: sim matrix implementation
        # sim = torch.arange(102, 102+args.num_sim).expand([len(data[0]), -1])
        # sim = torch.arange(109, 109+args.num_sim).expand([len(data[0]), -1])
        sim = torch.arange(113, 113+args.num_sim).expand([len(data[0]), -1])
        # sim[:, -1] = 101 # first [None of above] choice
        candidates = torch.cat(candidates, dim=0)
        scores = torch.cat(scores, dim=0)
        answers = torch.cat(answers, dim=0)
        
        hit_k = torch.mean((answers<args.k).float()).item()
        gnn_acc = torch.mean((answers==0).float()).item()

        print(f'{mode} hit-k({args.k}): {hit_k:.5f}')
        print(f'{mode} gnn accuracy: {gnn_acc:.5f}')
        
        print(answers[100:150])
        assert False

        acc = prompt_reasoning(sim, candidates, scores, answers, identifiers, args)
        print(f'{mode} gnn+llm accuracy: {acc:.5f}')
'''

def prompt_reasoning(sims, test_info, context_info, args):
    llm = LLM(args)
    sims = sims.tolist()
    
    # reaction infomation for test set
    candidates, scores, labels, identifiers = test_info
    reactant_identifiers = identifiers['reactant']
    product_identifiers = identifiers['product']
    corpus_identifiers = identifiers['corpus']
    candidates = candidates.tolist()
    scores = scores.tolist()
    answers = labels.tolist()

    # reaction infomation for in-context examples
    context_candidates, context_scores, context_labels, context_identifiers = context_info
    context_reactant_identifiers = context_identifiers['reactant']
    context_product_identifiers = context_identifiers['product']
    context_candidates = context_candidates.tolist()
    context_scores = context_scores.tolist()
    context_answers = context_labels.tolist()

    print(f'subset gnn acc: {torch.mean((labels[:subset_len]==0).float()).item():.5f}')
    print(f'subset gnn hit@{args.k}: {torch.mean((labels[:subset_len]<args.k).float()).item():.5f}')

    llm_results = []
    score_results = []
    labels = []
    prompt_history = []
    ans_history = []
    infer_time = []
    token_len = []
    if args.query_num >=10:
        mes_results = []
    for idx in tqdm(range(0, len(sims))):
        t_start = time.time()
        prompt_batch = []
        sim = sims[idx]
        # the text information of the question to be answered
        opt = [(corpus_identifiers['smiles'][j], corpus_identifiers['iupac'][j]) for j in candidates[idx]]
        score, answer = scores[idx], answers[idx]
        query_context = (
            [reactant_identifiers['smiles'][idx], reactant_identifiers['iupac'][idx]],
            opt,
            score,
            0,
            0
        )        
        labels.append(answer)
        for query_sim in divide_list(sim, args.num_sim):
            context = []
            # the text information of in-context examples
            for context_id, i in enumerate(query_sim):
                opt = [(context_product_identifiers['smiles'][j], context_product_identifiers['iupac'][j]) for j in context_candidates[i]]
                if args.shuffle_choice:
                    opt, score, answer = shuffle(opt, context_scores[i], context_answers[i])
                else:
                    score, answer = context_scores[i], context_answers[i]

                if context_id==args.num_sim-1 and args.conf:
                    # confidence = random.randint(8, 9)
                    confidence = random.randint(1, 1+args.rand_len)
                    answer = random_choice(len(opt), answer)
                else:
                    confidence = random.randint(9-args.rand_len, 9)
                example_context = (
                    [context_reactant_identifiers['smiles'][i], context_reactant_identifiers['iupac'][i]],
                    opt,
                    score,
                    answer,
                    confidence
                )
                context.append(example_context)
            context.append(query_context)
            
            # random.shuffle(context)
            if args.llm == 'gpt':
                if args.useJson:
                    prompt_batch.append(prompt_json(context, func=smilesFunc(args.smiles_wrap), useD=args.useD, conf=args.conf))
                else:
                    prompt_batch.append(prompt_chat(context, func=smilesFunc(args.smiles_wrap), useD=args.useD, conf=args.conf, name=args.mode))
                # prompt.append(prompt_chat_single(context, func=smilesFunc(args.smiles_wrap)))
            else:
                if args.useJson:
                    prompt_batch.append(prompt_json_in_context(context, func=smilesFunc(args.smiles_wrap), conf=args.conf))
                else:
                    prompt_batch.append(prompt_in_context(context, func=smilesFunc(args.smiles_wrap), conf=args.conf, name=args.mode))
        if args.useD or (answers[idx]<args.k):
            ans = llm(prompt_batch)
            ans_history.append(ans)
            if args.query_num >=10:
                ans = [query_essemble(ans[:q_num], naive_mes=args.noConf) for q_num in (1, 3, 5, 10)]
                mes_results.append([a[0] for a in ans])
                ans, score = ans[-1]
            else:
                ans, score = query_essemble(ans, naive_mes=args.noConf)
            print(to_char(ans), score, f'label: {to_char(labels[-1])}', sep='\t', flush=True)
            t_end = time.time()
            infer_time.append(t_end - t_start)
            token_len.append(token_count(prompt_batch, args.llm == 'gpt'))
        else:
            ans_history.append((0,0))
            ans, score = 0, 0
            if args.query_num >=10:
                mes_results.append([0 for _ in (1, 3, 5, 10)])

        prompt_history.append(prompt_batch[0])
        llm_results.append(ans)
        score_results.append(score)

    if args.query_num >= 10:
        mes_results = list(np.array(mes_results, dtype=np.int32).T)
    llm_results = np.array(llm_results, dtype=np.int32)
    score_results = np.array(score_results)
    labels = np.array(labels, dtype=np.int32)
    acc = (llm_results == labels).mean()
    
    print('llm answer:  ', ' '.join(map(to_char, llm_results.tolist())))
    print('llm scores:  ', ' '.join(map(str, score_results.tolist())))
    print('ground truth:', ' '.join(map(to_char, labels.tolist())))
    if args.query_num >=10:
        for mes_result, q_num in zip(mes_results, (1, 3, 5, 10)):
            print(f'gnn+llm accuracy (mes-{q_num}): {(mes_result==labels).mean():.5f}')
    print(f'gnn+llm accuracy: {acc:.5f}')
    print(f'Avg inference time:{avg(infer_time):.4f}s ({sum(infer_time):.2f}/{len(infer_time)})')
    print(f'Avg token cnt:{avg(token_len):.2f} ({sum(token_len)}/{len(token_len)})')

    # print_chat(prompt_history[-1])
    # for idx in np.where(np.logical_and(llm_results!=labels, labels!=args.k))[0].tolist():
    llm_correctness = list(llm_results==labels)
    gnn_correctness = list(labels==0)
    for idx in np.where(labels!=args.k)[0].tolist():
        print(f'\nreaction No.{idx}')
        if args.llm == 'gpt':
            print_chat(prompt_history[idx])
        else:
            print(prompt_history[idx])
        print('llm '+('correct' if llm_correctness[idx] else 'wrong'))
        print('gnn '+('correct' if gnn_correctness[idx] else 'wrong'))
        print(f'llm answer: {to_char(int(llm_results[idx]))}')
        print(f'ground truth: {to_char(int(labels[idx]))}')
        print(f'answer history:')
        for c, s in ans_history[idx]:
            print(f'{to_char(int(c))}\t{s}')
    return acc

    