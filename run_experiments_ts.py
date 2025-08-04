import data_manager, model_manager, model_inspector
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import checker
from tqdm import tqdm
import datasets
import pickle
import argparse
import os

def filename_getter(name, dataset, n_stories):

    if '/' in name:
        name = name.split('/')[1]

    if 'results' not in os.listdir():
        os.mkdir('results')
    
    return f'results/{name}_{dataset}_{n_stories}.pkl'

def args_parser():
    parser = argparse.ArgumentParser(description='Run experiments with a model')
    parser.add_argument('model_name', type=str, help='Model name')
    parser.add_argument('dataset', type=str, help='Datasets to use')
    parser.add_argument('--update_steps', type=int, default=10_000, help='Update steps')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--max_len', type=int, default=2, help='Max length')
    parser.add_argument('--n_stories', type=int, default=500, help='Number of stories to generate')
    return parser.parse_args()

def main(model_name, dataset, update_steps=10_000, device='cuda:0', max_len=2, n_stories=10):

    file_name = filename_getter(model_name, dataset, n_stories)
    mi = model_inspector.ModelInspector(model_name, dtype='float16', device=device)


    result_dict = {'activations': [], 'task': [], 'model': [], 'prompt': [], 'ansewer': []}

    dm = data_manager.DataManagerTs()
    
    pbar = tqdm(dm.prompt_generator(n_stories=n_stories, scramble=True, tasks='all', pre_append_prompt=True), total=(n_stories*(6*4)))
    i = 0
    for prompt, prompt_type in pbar:
        if len(prompt) > 1500:
            print(f'Skipping, len: {len(prompt)}')
            continue    

        txt, all_act = mi.generate_with_activations(prompt, max_len=max_len, verbose=False, sample=True)
        
        result_dict['activations'].append(all_act.cpu())
        result_dict['task'].append(dataset)
        result_dict['model'].append(model_name)
        result_dict['prompt'].append(prompt_type)
        result_dict['ansewer'].append(txt)


        if (i+1) % update_steps == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(result_dict, f)
        i += 1

    with open(file_name, 'wb') as f:
        pickle.dump(result_dict, f)

    return result_dict

if __name__ == '__main__':
    args = args_parser()
    res = main(args.model_name, args.dataset, args.update_steps, args.device, args.max_len, args.n_stories)
    ds = datasets.Dataset.from_dict(res)
    print(ds)
    ds.save_to_disk(f'results/{args.model_name}_{args.dataset}_{args.n_stories}')