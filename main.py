import os
import data
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, BertModel
# from sklearn.metrics import average_precision_score, label_ranking_average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import eval
import matplotlib.pyplot as plt
import warnings
import engine_train
import argparse
import yaml
import utils

warnings.filterwarnings("ignore")


def main(config):

    training_method = config.training_method
    save_path = config.save_path
    
    train_data_path = config.train_data_path
    test_data_path = config.test_data_path
    eval_data_path = config.eval_data_path

    num_task = config.num_task
    num_support = config.num_support
    num_query = config.num_query

    num_epochs = config.num_epochs
    lr = config.lr
    batch_size = config.batch_size

    train_dataset = data.wikiQA_dataset(data.load_data(train_data_path))
    test_dataset = data.wikiQA_dataset(data.load_data(test_data_path))
    eval_dataset= data.wikiQA_dataset(data.load_data(eval_data_path))

    # 加载BERT模型和分词器
    model_name = 'pretrained_models/bert-base-uncased'
    if training_method == 'pn':
        model = BertModel.from_pretrained(model_name)
    else:
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(model_name)


    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= num_epochs, eta_min=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_train_loss = []
    total_test_loss = []
    total_map = []
    total_mrr = []

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    tasks = train_dataset.pn_data_gen(num_task=num_task,num_support=num_support,num_query=num_query)

    best = 0 
    for epoch in range(num_epochs):
        if training_method == 'fullysup':
            train_loss= engine_train.fully_sup(model, tokenizer, train_loader,optimizer, device)
            map, mrr, test_loss = eval.test(model, tokenizer, test_loader, criterion, device,test_data_path)
        elif training_method == 'maml':
            train_loss= engine_train.maml_train(model, tokenizer, tasks, 1e-5, optimizer, 1, device)
            map, mrr, test_loss = eval.test(model, tokenizer, test_loader, criterion, device,test_data_path)
        elif training_method == 'pn':
            train_loss= engine_train.pn_train(model, tokenizer, tasks, optimizer, device,2)
            map, mrr, test_loss = eval.pn_test(model, tokenizer, tasks, test_dataset, device,test_data_path)
        elif training_method == 'reptile':
            train_loss= engine_train.reptile_train(model, tokenizer, tasks, 1e-5, optimizer, 5, device)
            map, mrr, test_loss = eval.test(model, tokenizer, test_loader, criterion, device,test_data_path)

        total_train_loss.append(train_loss)
        total_test_loss.append(test_loss)
        total_map.append(map)
        total_mrr.append(mrr)
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss}')
        print(f'Test Loss: {test_loss}')
        print('MAP: {}, MRR: {}'.format(map, mrr))
        print('-' * 40)
        with open(save_path+'log.txt','a+') as f:
            x = f'Epoch {epoch+1}/{num_epochs}' + '\t' + f'Train Loss: {train_loss}' + '\t' + f'Test Loss: {test_loss}' + '\t' +'MAP: {}, MRR: {}'.format(map, mrr) + '\n'
            f.write(x)
        plt.plot(total_train_loss)
        plt.savefig(save_path+'train_loss.png')
        plt.close()

        plt.plot(total_test_loss)
        plt.savefig(save_path+'test_loss.png')
        plt.close()

        plt.plot(total_map)
        plt.plot(total_mrr)
        plt.savefig( save_path+'map_mrr.png')
        plt.close()

        model.save_pretrained(save_path+'last_model')
        tokenizer.save_pretrained(save_path+'last_model')

        if (map+mrr) >= best:
            best = map+mrr
            model.save_pretrained(save_path+'best_model')
            tokenizer.save_pretrained(save_path+'best_model') 
    



if __name__ == '__main__':

    

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    utils.set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='experiments/wikiqa/config.yaml')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
        
    config = utils.DictToAttr(config_dict)

    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)

    with open(config.save_path + 'config.yaml', 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False,sort_keys=False)
        
    main(config)



    
    


    

    


