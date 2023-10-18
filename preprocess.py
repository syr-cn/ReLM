import pandas as pd
from utils import smilesFunc

def read_data(dataset, mode, func_name):
    func = smilesFunc(func_name)
    path = f'dataset/{dataset}/{mode}.csv'
    path_target = f'dataset/{dataset}/{mode}_{func_name}.csv'
    print('preprocessing %s data from %s' % (mode, path))

    # saving all possible values of each attribute (only for training data)
    data = []

    import time
    t1=time.time()
    with open(path) as f:
        for line in f.readlines():
            idx, product_smiles, reactant_smiles, _ = line.strip().split(',')

            if len(idx) == 0:
                continue
            if int(idx) % 10000 == 0:
                print('%dk' % (int(idx) // 1000))
            product_identifier = '|'.join([func(x) for x in product_smiles.split('.')])
            reactant_identifier = '|'.join([func(x) for x in reactant_smiles.split('.')])
            data.append([idx, product_smiles, reactant_smiles, product_identifier, reactant_identifier])

    df = pd.DataFrame(data, columns=[
        '',
        'product_smiles',
        'reactant_smiles',
        f'prod_{func_name}',
        f'reactant_{func_name}'
    ])
    df.to_csv(path_target, index=False)


if __name__=='__main__':
    read_data('USPTO-479k', 'test', 'iupac')