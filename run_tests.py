import pickle
import torch
import model_inspector, model_manager
import pandas as pd
from tqdm import tqdm
from utils import evaluate_model_overall, evaluate_model_by_subject
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


def load_data(filename):

    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            ds = pickle.load(f)
        ds['activations'] = torch.stack(ds['activations'])
    else:
        raise ValueError(f'Unknown file format: {filename}')
    return ds


def apply_to_batches(ds, f, bs=16):
    results = []
    for i in tqdm(range(0, ds['activations'].shape[0], bs)):
        results.append(f(ds['activations'][i:i+bs]))
    return results

def main(filepath, functions=['entropy'], bs=16, label='correct'):

    ds = load_data(filepath)
    modelname = ds['model'][0]
    print('Loaded:', filepath)

    mi = model_inspector.ModelInspector(modelname, dtype='float16')

    functions = {'entropy': lambda x: mi.calculate_entropy_batched(x.cuda(), num_windows=1)}
    if 'entropy' not in ds.keys():
        
        for f in functions:
            print('Calculating', f)
            results = apply_to_batches(ds, functions[f], bs=bs)
            ds[f] = torch.vstack(results).cpu().numpy()
            ds[f] = ds[f].reshape(ds['entropy'].shape[0], -1)
    
    X = ds['entropy']
    if label == 'correct':
        y = torch.tensor(ds[label])
    else:
        y = pd.Series(ds['prompt']).str.split('_').apply(lambda x: x[0]).astype('category').cat.codes.values
        y = torch.tensor(y)

    # clf = KNeighborsClassifier(n_neighbors=11, n_jobs=-1)
    res_overall = evaluate_model_overall(X, y, clf=None, n_splits=10)
    # res_by_subject = evaluate_model_by_subject(X, y, ds['subject'])
    res_by_subject = None

    # count percentage of correct answers
    res_overall['correct'] = y.float().mean().item()

    return res_overall, res_by_subject


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate model using entropy')
    parser.add_argument('filepath', type=str, help='Path to the data file')
    parser.add_argument('--bs', type=int, default=8, help='Batch size')
    parser.add_argument('--label', type=str, default='correct', help='Label to use')
    args = parser.parse_args()

    res_overall, res_by_subject = main(args.filepath, bs=args.bs, label=args.label)

    print('Overall results:')
    print(res_overall)
    print('Results by subject:')
    print(res_by_subject)

    res_file = args.filepath.replace('.pkl', '_tm_overall.csv')
    res_overall.to_csv(res_file, index=False)
    print('Results saved to', res_file)
    res_file = args.filepath.replace('.pkl', '_tm_subject.csv')
    # res_by_subject.to_csv(res_file, index=False)
    print('Results saved to', res_file)

    