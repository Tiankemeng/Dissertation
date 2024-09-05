import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random

def load_data(path):
    
    data = pd.read_csv(path, delimiter='\t')
    
    all_samples = []
    for i in range(len(data)):
        raw_data = data.iloc[i]
        sample = {}
        sample['Q_ID'] = raw_data['QuestionID']
        sample['question'] = raw_data['Question'].lower()
        sample['answer'] = raw_data['Sentence'].lower()
        sample['A_id'] = raw_data['SentenceID']
        sample['label'] = raw_data['Label']
        all_samples.append(sample)
    return all_samples


class wikiQA_dataset(Dataset):
    def __init__(self, data):
        self.raw_data = data
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        item = self.raw_data[idx]
        return item
    
    def task_gen(self,num_task,num_query,num_support):
        sample_idx = random.sample(range(0, len(self.raw_data)), num_task * (num_query+num_support))
        task = []
        for i in range(num_task):
            task_id = sample_idx[i* (num_query+num_support):(i+1)* (num_query+num_support)]
            supprot_set = [self.raw_data[task_id[0:num_support][i]] for i in range(len(task_id[0:num_support]))]
            supprot_set = wikiQA_dataset(supprot_set)
            query_set = [self.raw_data[task_id[num_support:-1][i]] for i in range(len(task_id[num_support:-1]))]
            query_set = wikiQA_dataset(query_set)
            task.append((supprot_set,query_set))
        return task
    
    def pn_data_gen(self,num_task,num_query,num_support):
        pos_set = []
        neg_set  = []
        for idx in range(len(self.raw_data)):
            item = self.raw_data[idx]
            if item['label'] == 0:
                neg_set.append(item)
            else:
                pos_set.append(item)
        # sample_idx = random.sample(range(0, len(self.raw_data)), num_task * (num_query+num_support))
        pos_sample_idx = random.sample(range(0, len(pos_set)),  num_task *(num_query+num_support))
        neg_sample_idx = random.sample(range(0, len(neg_set)),  num_task * (num_query+num_support))

        tasks = []
        for i in range(num_task): 
            pos_task_id = pos_sample_idx[i* (num_query+num_support):(i+1)* (num_query+num_support)]
            neg_task_id = neg_sample_idx[i* (num_query+num_support):(i+1)* (num_query+num_support)]
            
            supprot_set = [pos_set[pos_task_id[0:num_support][i]] for i in range(len(pos_task_id[0:num_support]))] \
                        + [neg_set[neg_task_id[0:num_support][i]] for i in range(len(neg_task_id[0:num_support]))]
            supprot_set = wikiQA_dataset(supprot_set)

            query_set = [pos_set[pos_task_id[num_support:len(pos_sample_idx)+1][i]] for i in range(len(pos_task_id[num_support:len(pos_sample_idx)+1]))] \
                        + [neg_set[neg_task_id[num_support:len(neg_sample_idx)+1][i]] for i in range(len(neg_task_id[num_support:len(neg_sample_idx)+1]))]
            query_set = wikiQA_dataset(query_set)

            tasks.append((supprot_set,query_set))
            
        return tasks
