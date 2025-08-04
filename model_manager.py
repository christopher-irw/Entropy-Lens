import pandas as pd
import os
URL_MODEL_TABLE = 'https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html'
MODEL_FAMILIES = ['gpt2', 'pythia', 'opt', 'bloom', 'phi', 'gemma', 'gemma-2', 'meta', 'llama-3', 'llama-3.1', 'llama-3.2', 'qwen', 'qwen3', 'qwen2.5']


def download_and_parse(filename):
    df = pd.read_html(URL_MODEL_TABLE)[0].rename(columns={'name.default_alias': 'model'})
    df['model_family'] = df['model'].apply(lambda x: x.split('-')[0])

    # parse pythia models
    to_drop = ['deduped', 'v0', 'seed']
    df = df[~df['model'].str.contains('|'.join(to_drop))]

    # parse gemma models
    # if act_fn is gelu_pytorch_tanh then it is a gemma-2 model otherwise it is a gemma model
    idx_gemma2 = df[(df['model_family'] == 'gemma') & (df['cfg.act_fn'] == 'gelu_pytorch_tanh')].index
    df.loc[idx_gemma2, 'model_family'] = 'gemma-2'
    df[df.model_family == 'gemma-2']

    # parse Llama-3 models
    for i,row in df.iterrows():
        if 'Meta-Llama-3' in row['model']:
            df.loc[i, 'model_family'] = 'llama-3'
        if 'Llama-3.1' in row['model']:
            df.loc[i, 'model_family'] = 'llama-3.1'
        if 'Llama-3.2' in row['model']:
            df.loc[i, 'model_family'] = 'llama-3.2'
    
    # TODO parse more model families: e.g. Llama-2
    df = df[df.model_family.isin(MODEL_FAMILIES)].reset_index(drop=True)

    # add model size as float
    df['model_size'] = 0.
    for i, row in df.iterrows():
        if 'M' in row['n_params.as_str']:
            df.loc[i, 'model_size'] = float(row['n_params.as_str'].split('M')[0])/1000
        elif 'B' in row['n_params.as_str']:
            df.loc[i, 'model_size'] = float(row['n_params.as_str'].split('B')[0])

    df.to_csv(filename, index=False)

    return df

def get_model_list(filename='model_properties.csv', max_size=8, model_families=MODEL_FAMILIES, update=False):
    if os.path.exists(filename) and not update:
        print('Reading model list from file')
        df = pd.read_csv(filename)
    else:
        print('Downloading and parsing model list')
        df = download_and_parse(filename)
    
    if len(model_families) > 0:
        df = df[df.model_family.isin(model_families)]
        df = df[df.model_size <= max_size].reset_index(drop=True)
    return df

    
if __name__ == '__main__':
    df = get_model_list(max_size=8, model_families=['gpt2'], update=True)
    print(df)
