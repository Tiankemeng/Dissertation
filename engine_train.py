
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import copy
import utils


warnings.filterwarnings("ignore")

# 训练和测试函数
def fully_sup(model,tokenizer, train_loader,optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader):
        inputs = tokenizer(batch['question'], batch['answer'], truncation=True, padding='max_length', max_length=512,return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def maml_train(model, tokenizer, tasks, inner_lr, outer_optimizer, num_inner_steps,device):
    
    total_loss = 0
    sum_gradients = []
    for task_id,(support_set, query_set) in enumerate(tasks):
        task_model = copy.deepcopy(model)
        task_optimizer = optim.Adam(task_model.parameters(), lr=inner_lr,weight_decay=1e-3)
        for step in range(num_inner_steps):
            for batch in DataLoader(support_set, batch_size=16, shuffle=True):
                inputs = tokenizer(batch['question'], batch['answer'], truncation=True, padding='max_length', max_length=512,return_tensors="pt")
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = task_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                task_optimizer.step()
                task_optimizer.zero_grad()

        # Outer loop: evaluate on query set
        for batch in DataLoader(query_set, batch_size=16, shuffle=True):
            inputs = tokenizer(batch['question'], batch['answer'], truncation=True, padding='max_length', max_length=512,return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = task_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            # task_model.to(torch.device('cpu'))
            for i, params in enumerate(task_model.parameters()):
                if task_id == 0:
                    sum_gradients.append(copy.deepcopy(params.grad))
                else:
                    sum_gradients[i] += copy.deepcopy(params.grad)
            total_loss += loss.item()
        del task_model, task_optimizer
        torch.cuda.empty_cache()
            
    for i in range(0,len(sum_gradients)):
        sum_gradients[i] = sum_gradients[i] / float(len(tasks))

    #Assign gradient for original model, then using optimizer to update its weights
    for i, params in enumerate(model.parameters()):
        params.grad = sum_gradients[i]
    outer_optimizer.step()
    outer_optimizer.zero_grad()
    del sum_gradients
    
    # total_loss.backward()  
    
    return total_loss




def pn_train(model, tokenizer, tasks, optimizer, device,num_classes):
    model.train()
    running_loss = 0.0

    for support_set, query_set in tasks:
        support_embeddings = []
        labels = []
        with torch.no_grad():
            for support_batch in DataLoader(support_set, batch_size=20, shuffle=False):
                support_inputs = tokenizer(support_batch['question'], support_batch['answer'], truncation=True, padding='max_length', max_length=512,return_tensors="pt")
                support_input_ids = support_inputs['input_ids'].to(device)
                support_mask   = support_inputs['attention_mask'].to(device)
                support_embeddings.append(model(support_input_ids,support_mask)[1])
                labels.append(support_batch['label'])
            support_embeddings = torch.cat(support_embeddings,dim = 0)
            labels = torch.cat(labels,dim = 0)

            prototypes = utils.compute_prototypes(support_embeddings, labels,num_classes=num_classes)

        for query_batch in DataLoader(query_set, batch_size=8, shuffle=False):
            query_inputs = tokenizer(query_batch['question'], query_batch['answer'], truncation=True, padding='max_length', max_length=512,return_tensors="pt")
            query_input_ids = query_inputs['input_ids'].to(device)
            query_mask   = query_inputs['attention_mask'].to(device)
            optimizer.zero_grad()
            query_embeddings = model(query_input_ids, query_mask)[1]
            distances = torch.cdist(query_embeddings, prototypes)
            labels = query_batch['label'].to(device)
            log_p_y = torch.nn.functional.log_softmax(-distances, dim=1)
            loss = -log_p_y.gather(1, labels.view(-1, 1)).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss 


def reptile_train(model, tokenizer, tasks, inner_lr, outer_optimizer, num_inner_steps,device):
    
    total_loss = 0
    sum_gradients = []
    for task_id,(support_set, query_set) in enumerate(tasks):
        task_model = copy.deepcopy(model)
        task_optimizer = optim.Adam(task_model.parameters(), lr=inner_lr,weight_decay=1e-3)
        for step in range(num_inner_steps):
            for batch in DataLoader(support_set, batch_size=16, shuffle=True):
                inputs = tokenizer(batch['question'], batch['answer'], truncation=True, padding='max_length', max_length=512,return_tensors="pt")
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = task_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                task_optimizer.step()
                task_optimizer.zero_grad()

        meta_weights = list(model.parameters())
        fast_weights = list(task_model.parameters())
    
        for i, (meta_params, fast_params) in enumerate(zip(meta_weights, fast_weights)):
            gradient = meta_params - fast_params
            if task_id == 0:
                sum_gradients.append(gradient)
            else:
                sum_gradients[i] += gradient

        # Outer loop: evaluate on query set
        with torch.no_grad():
            for batch in DataLoader(query_set, batch_size=16, shuffle=True):
                inputs = tokenizer(batch['question'], batch['answer'], truncation=True, padding='max_length', max_length=512,return_tensors="pt")
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = task_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
        
        del task_model, task_optimizer
        torch.cuda.empty_cache()
            
    for i in range(0,len(sum_gradients)):
        sum_gradients[i] = sum_gradients[i] / float(len(tasks))

    for i, params in enumerate(model.parameters()):
        params.grad = sum_gradients[i]
    outer_optimizer.step()
    outer_optimizer.zero_grad()
    
    del sum_gradients
    
    # total_loss.backward()  
    
    return total_loss

