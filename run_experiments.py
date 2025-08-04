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

def filename_getter(name, dataset, prompt):

    if '/' in name:
        name = name.split('/')[1]

    if 'results' not in os.listdir():
        os.mkdir('results')
    
    return f'results/{name}_{dataset}_{prompt}.pkl'

def args_parser():
    parser = argparse.ArgumentParser(description='Run experiments with a model')
    parser.add_argument('model_name', type=str, help='Model name')
    parser.add_argument('dataset', type=str, help='Datasets to use')
    parser.add_argument('--update_steps', type=int, default=10_000, help='Update steps')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--max_len', type=int, default=2, help='Max length')
    parser.add_argument('--prompt_type', type=str, default='base', help='Prompt type')
    return parser.parse_args()

def main(model_name, dataset, update_steps=10_000, device='cuda:0', max_len=2, prompt_type='base'):

    file_name = filename_getter(model_name, dataset, prompt_type)
    mi = model_inspector.ModelInspector(model_name, dtype='float16', device=device)


    # result_dict = {'activations': [], 'task': [], 'model': [], 'prompt': [], 'ansewer': [], 'subject': [], 'correct': []}
    result_dict = {'activations': [], 'task': [], 'model': [], 'prompt': [], 'subject': [], 'correct': []}

    dm = data_manager.DatasetManager()
    questions = dm.questions(dm.get_subjects(), random_state=55)
    pbar = tqdm(questions.iterrows(), total=questions.shape[0])
    for i, row in pbar:

        if len(row.question) > 1000:
            print(f'Skipping question: {i}, len: {len(row.question)}')
            continue    

        prompt = getattr(dm, f'get_{prompt_type}_prompt')(row)
        
        pbar.set_postfix({'len_prompt': len(prompt)})

        txt, all_act = mi.generate_with_activations(prompt, max_len=max_len, verbose=False)
        
        result_dict['activations'].append(all_act.cpu())
        result_dict['task'].append(dataset)
        result_dict['model'].append(model_name)
        result_dict['prompt'].append(prompt_type)
        # result_dict['ansewer'].append(txt)
        result_dict['subject'].append(row.subject)
        result_dict['correct'].append(checker(txt, row.answer))


        if (i+1) % update_steps == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(result_dict, f)

    with open(file_name, 'wb') as f:
        pickle.dump(result_dict, f)

    return result_dict

if __name__ == '__main__':
    args = args_parser()
    res = main(args.model_name, args.dataset, args.update_steps, args.device, args.max_len, args.prompt_type)
    # ds = datasets.Dataset.from_dict(res)
    # print(ds)
    # ds.save_to_disk(f'results/{args.model_name}_{args.dataset}_{args.prompt_type}')
