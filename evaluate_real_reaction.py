import numpy as np
from scipy.spatial.distance import cdist
from model import load_featurizer
import pandas as pd
from prompt_generator import prompt_single
from utils import smiles2name

def evaluate_real_reaction(args):
    model = load_featurizer(args)
    df = pd.read_csv('dataset/real_reaction_test/real_reaction_test.csv')
    num_choise = 5

    for idx, row in df.iterrows():
        if idx == 0:
            continue
        reactant = row['reactant']
        options = [row[f'choice_{i}'] for i in range(num_choise)]
        answer = row['answer']
        
        r_emb, _ = model.transform([reactant])
        p_embs, _ = model.transform(options)
        dist = cdist(r_emb, p_embs, metric='euclidean')[0]
        choices, dists, answer = election(options, dist, answer)

        query = prompt_single(reactant, choices, dists, answer, smiles2name)
        print(query)


def election(samples, dist, answer, num=3, reverse=False):
    # set reverse=True to sort in decreasing order
    # sorting in increasing order by default
    sorted_indices = list(np.argsort(dist))
    if reverse:
        sorted_indices.reverse()

    if answer in sorted_indices[:num]:
        sorted_indices = sorted_indices[:num]
    else:
        sorted_indices = sorted_indices[:num-1] + [answer]
    
    np.random.shuffle(sorted_indices)
    samples_ = [samples[i] for i in sorted_indices]
    dist_ = [dist[i] for i in sorted_indices]
    answer_ = sorted_indices.index(answer)

    return samples_, dist_, answer_


if __name__=='__main__':
    evaluate_real_reaction()