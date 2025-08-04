import os
import pandas as pd
import datasets
import numpy as np

class DatasetManager():
    def __init__(self, data_dir_path='./data'):
        self.splits = {'test': 'hf://datasets/cais/mmlu/all/test-00000-of-00001.parquet', 
                       'validation': 'hf://datasets/cais/mmlu/all/validation-00000-of-00001.parquet', 
                       'dev': 'hf://datasets/cais/mmlu/all/dev-00000-of-00001.parquet', 
                       'auxiliary_train': 'hf://datasets/cais/mmlu/all/auxiliary_train-00000-of-00001.parquet'
                      }
        self.data_dir_path = data_dir_path
        self.data = self.load_data('test')
        
    def load_data(self, split):
        if not os.path.exists(self.data_dir_path):
            os.makedirs(self.data_dir_path)

        if os.listdir(self.data_dir_path) == []:
            df = pd.read_parquet(self.splits[split])
            df.to_parquet(self.data_dir_path + f'/mmlu_{split}.parquet')
        else:
            df = pd.read_parquet(self.data_dir_path + f'/mmlu_{split}.parquet')
        return df

    def get_instruct_prompt(self, row):
        prompt = f'The following is a multiple choice question about {row.subject}. Reply only with the correct option.\n\n'
        prompt += f'Question: {row.question}\n\n'
        prompt += f'Choices:\n'
        for i, choice in enumerate(row.choices):
            prompt += f'{chr(65+i)}. {choice}\n'
        prompt += '\nAnswer: '
        return prompt

    def get_thinking_prompt(self, row):
        prompt = f'The following is a multiple choice question about {row.subject}. Reply with a short 1 phrase explaination THEN the correct option.\n\n'
        prompt += f'Question: {row.question}\n\n'
        prompt += f'Choices:\n'
        for i, choice in enumerate(row.choices):
            prompt += f'{chr(65+i)}. {choice}\n'
        prompt += 'Explaination: '
        return prompt

    def get_base_prompt(self, row):
        prompt = f'Subject: {row.subject}.\n\n'
        prompt += f'Question: {row.question}\n\n'
        prompt += f'Choices:\n'
        for i, choice in enumerate(row.choices):
            prompt += f'{chr(65+i)}. {choice}\n'
        prompt += '\nAnswer: '
        return prompt
    
    def get_humble_prompt(self, row):
        prompt = f'The following is a multiple choice question about {row.subject}. Reply only with the correct option, if you are unsure about the answer, reply with a completly random option.\n\n'
        prompt += f'Question: {row.question}\n\n'
        prompt += f'Choices:\n'
        for i, choice in enumerate(row.choices):
            prompt += f'{chr(65+i)}. {choice}\n'
        prompt += '\nAnswer: '
        return prompt
    
    def get_subjects(self):
        return self.data.subject.unique()
    
    # def questions(self, subject, num_questions=None, random_state=42):
    #     subset = self.data[self.data.subject == subject]
    #     if num_questions:
    #         subset = subset.sample(num_questions, random_state=random_state)
    #     for i,row in subset.iterrows():
    #         yield self.get_prompt(row)

    def questions(self, subject, num_questions=None, random_state=42):
        # subset = self.data[self.data.subject == subject]
        subset = self.data[self.data.subject.isin(subject)]
        if num_questions:
            subset = subset.sample(num_questions, random_state=random_state)
        return subset


class DataManagerTs():
    """Data manager for the tiny stories dataset"""
    def __init__(self, split='validation'):
        self.data = datasets.load_dataset('roneneldan/TinyStories', split='validation')
        self.data.to_pandas()

        self.tasks = {'continuation_prompt_1': 'How can the story be continued?',
                        'continuation_prompt_2': 'Which could be a continuation of the story?',
                        'counting_prompt_1': 'How many words are in the story?',
                        'counting_prompt_2': 'Count the number of words in the story.',
                        'semantic_prompt_1': 'What is the main idea of the story?',
                        'semantic_prompt_2': 'Who is the subject of the story?',
        }

    def scramble_story(self, story):
        words = story.split(' ')
        np.random.shuffle(words)
        return ' '.join(words)

    def prompt_generator(self, n_stories=100, scramble=False, tasks='all', pre_append_prompt=False):
        if tasks == 'all':
            tasks = self.tasks.keys()
        
        for i in range(n_stories):
            story = self.data['text'][i]
            for task in tasks:

                # base
                yield f'{story}\n\n{self.tasks[task]}\n\n', f'{task}_base'

                if pre_append_prompt:
                    yield f'{self.tasks[task]}\n\n{story}\n\n', f'{task}_rev'

                if scramble:
                    story = self.scramble_story(story)

                    yield f'{story}\n\n{self.tasks[task]}\n\n', f'{task}_scramble'

                    if pre_append_prompt:
                        yield f'{self.tasks[task]}\n\n{story}\n\n', f'{task}_scramble_rev'

if __name__ == '__main__':
    # dm = DatasetManager()
    # print(dm.get_subjects())
    # for q in dm.questions('abstract_algebra', num_questions=2):
    #     print(q)

    dmts = DataManagerTs()
    for p in dmts.prompt_generator(n_stories=1, scramble=False, tasks='all', pre_append_prompt=False):
        print(p)
        print('---'*8)