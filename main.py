import os
import yaml
import argparse
import data_processing
from evaluate_real_reaction import evaluate_real_reaction
from evaluate_gnn import evaluate_gnn
from evaluate_llm import evaluate_llm
from utils import set_seed
import train


def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='the index of gpu device')
    parser.add_argument('--seed', type=int, default=0, help='seed of model & choice shuffling')
    parser.add_argument('--dataset', type=str, default='USPTO-479k', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--task', type=str, default=None, help='downstream task')
    parser.add_argument('--model', type=str, default='MolR', help='model name')
    parser.add_argument('--smiles_wrap', type=str, default=None, help='wrapper function of smiles strings')
    parser.add_argument('--mode', type=str, default='test', help='subset name')
    parser.add_argument('--context', type=str, default=None, help='subset name')
    parser.add_argument('--corpus', type=str, default=None, help='allowable product list')
    parser.add_argument('--model_path', type=str, default=None, help='path of pretrained model')
    parser.add_argument('--config_path', type=str, default=None, help='path of pretrained model config')

    # pretraining setting
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=20, help='training epochs')
    parser.add_argument('--margin', type=float, default=4.0, help='margin in contrastive loss')

    # LLM & prompt settings
    parser.add_argument('--num_sim', type=int, default=2, help='few-shot example number')
    parser.add_argument('--llm', type=str, default='galactica', help='llm name')
    parser.add_argument('--k', type=int, default=3, help='candidate number')
    parser.add_argument('--rand_len', type=int, default=2, help='candidate number')
    parser.add_argument('--llm_bs', type=int, default=1, help='batch size')
    parser.add_argument('--max_tokens', type=int, default=10, help='max output token num for llms')
    parser.add_argument('--shuffle_choice', action='store_true', default=False)
    parser.add_argument('--useD', action='store_true', default=False)
    parser.add_argument('--useJson', action='store_true', default=False)
    parser.add_argument('--noConf', action='store_true', default=False)
    parser.add_argument('--openai_key', type=str, default=None, help='openai key')
    parser.add_argument('--gpt_temp', type=float, default=0, help='temperature param in openAi messages')
    parser.add_argument('--query_num', type=int, default=1, help='number of query per sample, used in multiquery ensemble strategy')

    args = parser.parse_args()
    args.conf = not args.noConf
    if not args.config_path:
        args.config_path = f'models/{args.model}/config.yaml'
    with open(args.config_path) as f:
        args.config = yaml.safe_load(f)
    if args.model_path:
        args.config['model_file'] = args.model_path

    print_setting(args)
    print('current working directory: ' + os.getcwd() + '\n')
    set_seed(args.seed)

    if args.task == 'train':
        train.train(args)
    elif args.task == 'evaluate':
        data = data_processing.load_data(args)
        evaluate_gnn(args, data)
    elif args.task == 'evaluate_llm':
        # data, smiles = data_processing.load_data_with_smiles(args)
        # evaluate_llm(args, data, smiles)
        evaluate_llm(args)
    else:
        raise ValueError('unknown task')


if __name__ == '__main__':
    main()
