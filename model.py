import os
import sys
import torch
from transformers import pipeline
from utils import timeit
import time
import openai
from models import *
import json

def load_featurizer(args):
    model_file = args.config['model_file']
    if args.model == 'MolR':
        model = MolEFeaturizer(args, args.device)
    else:
        raise NotImplementedError('Model path not supported!')
    return model

def load_model(args, load=True):
    model_file = args.config['model_file']

    if args.model == 'MolR':
        model = MolRGNN(args.config, device = args.device)
        model_file = os.path.join(model_file, 'model.pt')
    elif args.model =='LocalRetro':
        exp_config = args.config
        model = LocalRetro(
            node_in_feats=exp_config['in_node_feats'],
            edge_in_feats=exp_config['in_edge_feats'],
            node_out_feats=exp_config['node_out_feats'],
            edge_hidden_feats=exp_config['edge_hidden_feats'],
            num_step_message_passing=exp_config['num_step_message_passing'],
            attention_heads = exp_config['attention_heads'],
            attention_layers = exp_config['attention_layers'],
            AtomTemplate_n = exp_config['AtomTemplate_n'],
            BondTemplate_n = exp_config['BondTemplate_n'],
            device = args.device
        )
    elif args.model =='Megan':
        exp_config = args.config
        model = load_megan(
            save_path=args.config['save_path'],
            device=args.device
        )
    else:
        raise NotImplementedError('Model path not supported!')
    
    # loading model from file
    if load:
        assert os.path.isfile(model_file), f'<{model_file}> not found'
        model_param = torch.load(model_file, map_location=torch.device(args.device))
        if 'model_state_dict' in model_param:
            model_param = model_param['model_state_dict']
        model.load_state_dict(model_param)
        print(f'load model from {model_file}')
    model = model.to(args.device)

    return model


class LLM():
    def __init__(self, args):
        self.llm = args.llm.lower()
        self.device = args.device
        if self.llm == 'gpt':
            assert args.openai_key != None
            openai.api_key = args.openai_key
            self.gpt_temp = args.gpt_temp
        elif self.llm == 'galactica':
            self.task = "text-generation"
            model_name = "facebook/galactica-30b"
            self.model = pipeline(self.task, model=model_name, device=self.device)
        elif self.llm == 'opt':
            self.task = "text-generation"
            model_name = "facebook/opt-6.7b"
            self.model = pipeline(self.task, model=model_name, device=self.device)
        elif self.llm == 'vicuna':
            self.task = "text2text-generation"
            model_name = "lmsys/fastchat-t5-3b-v1.0"
            self.model = pipeline(self.task, model=model_name, device=self.device)
        else:
            raise NotImplementedError('LLM not supported!')
        self.max_tokens = args.max_tokens
        self.useJson = args.useJson

    # @timeit
    def __call__(self, query):
        if self.llm == 'gpt':
            # open Ai has rate limit, no parallelization here
            answer = []
            for q in query:
                response = self.gpt3_infer(q)
                answer.append(self.parse_gpt(response))
        elif self.task == "text-generation":
            answer = self.model(query, max_new_tokens=self.max_tokens)
            answer = [ans['generated_text'] for ans in answer]
            answer = [self.parse_answer_with_score(a[len(q):]) for q,a in zip(query, answer)]
        elif self.task == "text2text-generation":
            answer = self.model(query, max_new_tokens=self.max_tokens)
            answer = [ans['generated_text'] for ans in answer]
            answer = [self.parse_answer_with_score(a) for a in answer]
        return answer

    def gpt3_infer(self, query, retry_=0):
        if retry_ > 0:
            print('retrying...', file=sys.stderr)
            st = 2 ** retry_
            time.sleep(st)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=query,
                temperature=self.gpt_temp,
                max_tokens=self.max_tokens,
                stop=['<|endoftext|>']
            )
        except Exception as e:
            print(type(e), file=sys.stderr)
            if str(e) == 'You exceeded your current quota, please check your plan and billing details.':
                exit(1)
            return self.gpt3_infer(query, retry_ + 1)
        return response

    def parse_json(self, ans):
        try:
            ans = ans.replace('\t', '')
            ans = json.loads(ans)
            assert 'answer' in ans
            ans = ans['answer']
            
            assert 'choice' in ans
            assert 'confidence' in ans
            choice, score = ans['choice'], ans['confidence']
            return ord(choice)-ord('A'), int(score)
        except:
            print('error occured when parsing json')
            print(ans)
            return 0, 0

    def parse_answer_with_score(self, ans):
        try:
            ans = ans.strip()
            for c in '\n\r\t':
                ans = ans.replace(c, ' ')
            ans = ans.split(' ')

            choice, score = 0, 1
            for c in ans:
                c=''.join([i for i in c if i.isalnum()])
                if len(c)==0 or len(c) >= 2:
                    continue
                if 'A' <= c <= 'Z':
                    choice = c
                elif '0' <= c <= '9':
                    score = c
            return ord(choice)-ord('A'), int(score)
        except:
            print(f'{ans} is not a valid input!')
            return 0, 2

    
    def parse_gpt(self, response):
        ans = response['choices'][0]['message']['content']
        
        if self.useJson:
            ans, score = self.parse_json(ans)
        else:
            ans, score = self.parse_answer_with_score(ans)
        return ans, score