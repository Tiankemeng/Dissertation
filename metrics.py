import sys


def eval_map_mrr(answer_file, gold_file):
    dic = {}
    fin = open(gold_file)
    for line in fin:
        line = line.strip()
        if not line:
            continue
       
        cols = line.split('\t')
        
            
        if 'QuestionID' in cols[0]:
            continue

        q_id = cols[0]
        a_id = cols[4]

        if not q_id in dic:
            dic[q_id] = {}
        dic[q_id][a_id] = [cols[6], -1]
    fin.close()

    fin = open(answer_file)
    for line in fin:
        line = line.strip()
        if not line:
            continue
        
        cols = line.split('\t')
        
        q_id = cols[0]
        a_id = cols[1]
        rank = int(cols[2])
        dic[q_id][a_id][1] = rank
    fin.close()

    MAP = 0.0
    MRR = 0.0
    for q_id in dic:
        sort_rank = sorted(dic[q_id].items(), key=lambda asd: asd[1][1], reverse=False)
        correct = 0
        total = 0
        AP = 0.0
        mrr_mark = False
        for i in range(len(sort_rank)):
            # compute MRR
            if sort_rank[i][1][0] == '1' and mrr_mark == False:
                MRR += 1.0 / float(i + 1)
                mrr_mark = True
            # compute MAP
            total += 1
            if sort_rank[i][1][0] == '1':
                correct += 1
                AP += float(correct) / float(total)
        AP /= float(correct)
        MAP += AP

    MAP /= float(len(dic))
    MRR /= float(len(dic))
    return MAP, MRR

if __name__ == '__main__':
    answer_file = 'rank'
    gold_file = 'test.tsv'
    MAP, MRR = eval_map_mrr(answer_file,gold_file)
    print('MAP: {}, MRR: {}'.format(MAP, MRR))

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from transformers import BertTokenizer, BertForSequenceClassification
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    dataset = load_dataset("wiki_qa", data_dir='datasets/wikiqa')
    tokenizer = BertTokenizer.from_pretrained('bert_wikiqa_maml')
    model = BertForSequenceClassification.from_pretrained('bert_wikiqa_maml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    all_logits = []
    all_pred = []
    x = 'Q0\tD0-0\t0\n'
    for batch in DataLoader(dataset['test'], batch_size=16, shuffle=False):
        inputs = tokenizer(batch['question'], batch['answer'], truncation=True, padding='max_length', max_length=512,return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = batch['label'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_pred.append(preds)
    print(all_pred)