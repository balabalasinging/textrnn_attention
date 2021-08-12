import os
import random
import pandas as pd

data_list = []
file_dir = "data/"
all_csv_list = os.listdir(file_dir)
for single_csv in all_csv_list:
    with open(os.path.join(file_dir, single_csv),encoding='utf-8') as file:
        for line in file:
            label = line.split(',',3)[1]
            if(int(label)==0):
                label = 0
            if(int(label)==1):
                label = 1
            sentence = line.split(',',3)[2]
            data_list.append([sentence,label])

random.shuffle(data_list)
# 将全部语料按1:1:8分为测试集，验证集与训练集
n = len(data_list) // 10
dev_list = data_list[n:n*2]
train_list = data_list[n*2:]
test_list = data_list[:n]

print('训练集数量： {}'.format(str(len(train_list))))
print('测试集数量： {}'.format(str(len(test_list))))
print('验证集数量： {}'.format(str(len(dev_list))))
name=['Sentence','Label']
csv_train=pd.DataFrame(columns=name,data=train_list)
csv_train.to_csv('dataset/csv_train.csv',encoding='utf8',index=False)
csv_train=pd.DataFrame(columns=name,data=test_list)
csv_train.to_csv('dataset/csv_test.csv',encoding='utf8',index=False)
csv_train=pd.DataFrame(columns=name,data=dev_list)
csv_train.to_csv('dataset/csv_dev.csv',encoding='utf8',index=False)

