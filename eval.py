import numpy as np
import os
import data
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import metrics

def test(model, tokenizer, test_loader, criterion, device,label_path):
    model.eval()
    running_loss = 0.0
    all_output = {}
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = tokenizer(batch['question'], batch['answer'], truncation=True, padding='max_length', max_length=512,return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()
            logits = outputs.logits
            x = 'Q0\tD0-0\t0\n'
            for i in range(len(batch['question'])):
                if batch['Q_ID'][i] not in all_output:
                    all_output[batch['Q_ID'][i]] = [[batch['A_id'][i],logits[i][0].item()]]
                else:
                    all_output[batch['Q_ID'][i]].append([batch['A_id'][i],logits[i][0].item()])
    out_rank = []
    for q_id,answer in all_output.items():
        rank = utils.rank_elements([answer[i][-1] for i in range(len(answer))])
        rank = [i-1 for i in rank]
        for i in range(len(rank)):
            out_rank.append(q_id+'\t'+answer[i][0]+'\t'+str(rank[i])+'\n')
    with open('tmp.rank','w') as f:
        f.writelines(out_rank)
    map, mrr = metrics.eval_map_mrr('tmp.rank', label_path)
    return map, mrr,running_loss / len(test_loader),


def pn_test(model, tokenizer, tasks,test_loader, device,label_path):
    model.eval()
    running_loss = 0.0
    all_output = {}
    all_support_embedding = []
    all_support_label = []
    with torch.no_grad():
        for support_set, query_set in tasks:
            for support_batch in DataLoader(support_set, batch_size=20, shuffle=False):
                support_inputs = tokenizer(support_batch['question'], support_batch['answer'], truncation=True, padding='max_length', max_length=512,return_tensors="pt")
                support_input_ids = support_inputs['input_ids'].to(device)
                support_mask   = support_inputs['attention_mask'].to(device)
                support_embeddings = model(support_input_ids,support_mask)[1]
                all_support_embedding.append(support_embeddings)
                all_support_label.append(support_batch['label'])
        all_support_embedding = torch.cat(all_support_embedding,dim=0)
        all_support_label = torch.cat(all_support_label,dim = 0)
        prototypes = utils.compute_prototypes(all_support_embedding, all_support_label ,num_classes=2)

        for batch in DataLoader(test_loader, batch_size=16, shuffle=False):
            inputs = tokenizer(batch['question'], batch['answer'], truncation=True, padding='max_length', max_length=512,return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            mask  = inputs['attention_mask'].to(device)
            embeddings = model(input_ids,mask)[1]
            distances = torch.cdist(embeddings, prototypes)
            labels = batch['label'].to(device)
            log_p_y = torch.nn.functional.log_softmax(-distances, dim=1)
            loss = -log_p_y.gather(1, labels.view(-1, 1)).mean()
            running_loss+=loss.item()
            logits = log_p_y
            for i in range(len(batch['question'])):
                if batch['Q_ID'][i] not in all_output:
                    all_output[batch['Q_ID'][i]] = [[batch['A_id'][i],logits[i][0].item()]]
                else:
                    all_output[batch['Q_ID'][i]].append([batch['A_id'][i],logits[i][0].item()])
    out_rank = []
    for q_id,answer in all_output.items():
        rank = utils.rank_elements([answer[i][-1] for i in range(len(answer))])
        rank = [i-1 for i in rank]
        for i in range(len(rank)):
            out_rank.append(q_id+'\t'+answer[i][0]+'\t'+str(rank[i])+'\n')
    with open('tmp.rank','w') as f:
        f.writelines(out_rank)
    map, mrr = metrics.eval_map_mrr('tmp.rank', label_path)
    # map,mrr = 0,0
    return map, mrr,running_loss / len(test_loader),


if __name__ == '__main__':


    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # train_dataset = data.wikiQA_dataset(data.load_data('data/wikiQA/WikiQA-train.tsv'))
    test_dataset = data.wikiQA_dataset(data.load_data('data/wikiQA/WikiQA-test.tsv'))
    eval_dataset= data.wikiQA_dataset(data.load_data('data/wikiQA/WikiQA-dev.tsv'))

    # 加载BERT模型和分词器
    model_name = 'pretrained_models/bert-base-uncased'
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
    map, mrr, test_loss = test(model, tokenizer, eval_loader,  device,'data/wikiQA/WikiQA-dev.tsv')
    print('eavl result:\t', 'MAP: {}, MRR: {}'.format(map, mrr) )
    map, mrr, test_loss = test(model, tokenizer, test_loader,  device,'data/wikiQA/WikiQA-test.tsv')
    print('test result:\t', 'MAP: {}, MRR: {}'.format(map, mrr) )
