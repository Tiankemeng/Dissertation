import matplotlib.pyplot as plt
import os

def data_process(path):
    with open(path,'r') as f:
        raw_data = f.readlines()
    data= {}
    data['train_loss'] = []
    data['test_loss'] = []
    data['map'] = []
    data['mrr'] = []
    for i in raw_data:
        i = i.replace('\n','')
        sample = i.split('\t')
        # print(sample)
        data['train_loss'].append(float(sample[1].replace('Train Loss: ','')))
        data['test_loss'].append(float(sample[2].replace('Test Loss: ','')))
        data['map'].append(float(sample[3].split(',')[0].replace('MAP: ','')))
        data['mrr'].append(float(sample[3].split(',')[1].replace(' MRR: ','')))
    return data


def plot(pn_data,maml_data,reptile_data,name,save_path):
    
    # 示例数据
    
    pn_data = pn_data[name]
    maml_data = maml_data[name]
    reptile_data = reptile_data[name]
    x = [i for i in range(len(pn_data))]
    # 创建图形和轴
    fig, ax = plt.subplots()

    # 绘制折线图
    ax.plot(x, pn_data, label='pn')
    ax.plot(x, maml_data, label='maml')
    ax.plot(x, reptile_data, label='reptile')

    # 添加图例
    ax.legend()

    # 添加标题和轴标签
    ax.set_title(name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(name)

    # 显示图形
    plt.savefig(save_path+name+'.png')
    plt.close()


save_path = 'results/lr_1e-5_ep50_s20/summry/'

if not os.path.exists(save_path):
    os.mkdir(save_path)


dirname = 'results/lr_1e-5_ep50_s20/'
pn_data = dirname + 'wiki_pn/log.txt'
maml_data = dirname + 'wiki_maml/log.txt'
reptile_data = dirname + 'wiki_reptile/log.txt'

pn_data = data_process(pn_data)
maml_data = data_process(maml_data)
reptile_data = data_process(reptile_data)

save_path = 'results/lr_1e-5_ep50_s20/summry/wiki/'

if not os.path.exists(save_path):
    os.mkdir(save_path)


plot(pn_data,maml_data,reptile_data,'train_loss',save_path)
plot(pn_data,maml_data,reptile_data,'test_loss',save_path)
plot(pn_data,maml_data,reptile_data,'map',save_path)
plot(pn_data,maml_data,reptile_data,'mrr',save_path)


dirname = 'results/lr_1e-5_ep50_s20/'
pn_data = dirname + 'ip_pn/log.txt'
maml_data = dirname + 'ip_maml/log.txt'
reptile_data = dirname + 'ip_reptile/log.txt'

pn_data = data_process(pn_data)
maml_data = data_process(maml_data)
reptile_data = data_process(reptile_data)

save_path = 'results/lr_1e-5_ep50_s20/summry/ip/'

if not os.path.exists(save_path):
    os.mkdir(save_path)

plot(pn_data,maml_data,reptile_data,'train_loss',save_path)
plot(pn_data,maml_data,reptile_data,'test_loss',save_path)
plot(pn_data,maml_data,reptile_data,'map',save_path)
plot(pn_data,maml_data,reptile_data,'mrr',save_path)