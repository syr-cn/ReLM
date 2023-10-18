import pubchempy as pcp
import time
import functools
import random
import torch
import numpy as np
from collections import defaultdict
import tiktoken

def to_char(x):
    return chr(ord('A')+x)

def avg(x):
    return sum(x)/len(x)

def timeit(func):
    @functools.wraps(func)
    def timed(*args, **kw):
        ts = time.time()
        print(f'Method: {func.__name__} started!')
        result = func(*args, **kw)
        te = time.time()
        print(f'Method: {func.__name__} cost {te-ts:.2f} sec!')
        return result
    return timed

def divide_list(x, n):
    for i in range(0, len(x), n):
        yield x[i:i+n]

def query_essemble(ans, naive_mes=False):
    if len(ans)==1:
        return ans[0]
    result = defaultdict(list)
    for choice, score in ans:
        result[choice].append(score)
    if naive_mes:
        choice = max(result, key=lambda x: len(result[x]))
    else:
        choice = max(result, key=lambda x: avg(result[x]))
    score = int(avg(result[choice]))
    return choice, score

token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokens_per_gpt_message = 4
tokens_per_gpt_name = -1
# reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def token_count(text, gpt_msg=False):
    num_tokens = 0
    if gpt_msg:
        for chat in text:
            for message in chat:
                num_tokens += tokens_per_gpt_message
                for key, value in message.items():
                    num_tokens += len(token_encoder.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_gpt_name
    else:
        for string in text:
            num_tokens += len(token_encoder.encode(string))
    return num_tokens

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def smilesFunc(name):
    name = name.lower()
    if name == 'iupac':
        return smiles2iupac
    if name == 'iupac_smiles':
        return smiles2iupac
    elif name == 'name':
        return smiles2name
    elif name == 'galactica':
        return galactica_smiles
    return lambda x:x

def smiles2name(smiles):
    try:
        compound = pcp.get_compounds(smiles, 'smiles')[0]
        name = compound.synonyms[0]
        assert type(name)==str
        return name
    except:
        return smiles

def smiles_iupac(smiles):
    try:
        compound = pcp.get_compounds(smiles, 'smiles')[0]
        name = compound.iupac_name
        assert type(name)==str
        return f'{name}<{smiles}>'
    except:
        return smiles

# @timeit
def smiles2iupac(smiles):
    if '.' in smiles:
        return '|'.join([smiles2iupac(s) for s in smiles.split('.')])
    try:
        compound = pcp.get_compounds(smiles, 'smiles')[0]
        name = compound.iupac_name
        assert type(name)==str
        return name
    except:
        return smiles

def galactica_smiles(smiles):
    smiles = f'[START_I_SMILES]{smiles}[END_I_SMILES]'
    return smiles


def random_choice(n, i):
    try:
        index = list(range(n))
        index.remove(i)
        return random.choice(index)
    except:
        return random.randint(0, n-1)

def shuffle(option, score, answer):
    assert len(option) == len(score)
    assert answer <= len(option)
    n = len(option)

    indices = list(range(n))
    random.shuffle(indices)

    option = [option[i] for i in indices]
    score = [score[i] for i in indices]
    if answer < n:
        answer = indices.index(answer)
    return option, score, answer

def print_chat(message):
    print('\n')
    for dict_ in message:
        print(f"<role: {dict_['role']}>")
        print(dict_['content'])
    print('\n')