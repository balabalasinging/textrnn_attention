from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import DataSet


###########################
#使用TextRNN_Attention模型进行训练

import torch
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
import DataSet
import torch.nn as nn
import torch.nn.functional as F

class TextRNN_Attention(nn.Module):
    def __init__(self):
        super(TextRNN_Attention, self).__init__()
        Vocab = len(DataSet.getTEXT().vocab)  ## 已知词的数量
        Dim = 100  ##每个词向量长度
        dropout = 0.5
        hidden_size = 256 #隐藏层数量
        num_classes = 3 ##类别数
        num_layers = 2 ##双层LSTM
        hidden_size2 = 64

        self.embedding = nn.Embedding(Vocab, Dim)  ## 词向量，这里直接随机
        self.lstm = nn.LSTM(Dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size2)
        self.fc = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        # [batch len, text size]
        x = self.embedding(x)
        # [batch size, text size, embedding]
        output, (hidden, cell) = self.lstm(x)
        # output = [batch size, text size, num_directions * hidden_size]
        M = self.tanh1(output)
        # [batch size, text size, num_directions * hidden_size]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # [batch size, text size, 1]
        out = output * alpha
        # [batch size, text size, num_directions * hidden_size]
        out = torch.sum(out, 1)
        # [batch size, num_directions * hidden_size]
        out = F.relu(out)
        # [batch size, num_directions * hidden_size]
        out = self.fc1(out)
        # [batch size, hidden_size2]
        out = self.fc(out)
        # [batch size, num_classes]
        return out

###########################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model(test_iter, model, device):
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    accuracy = 0
    y_true = []
    y_pred = []
    total_test_num = len(test_iter.dataset)
    for batch in test_iter:
        feature = batch.text
        target = batch.label
        with torch.no_grad():
            feature = torch.t(feature)
        feature, target = feature.to(device), target.to(device)
        out = model(feature)
        loss = F.cross_entropy(out, target)
        total_loss += loss.item()
        accuracy += (torch.argmax(out, dim=1)==target).sum().item()
        y_true.extend(target.cpu().numpy())
        y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
    print('>>> Test loss:{}, Accuracy:{} \n'.format(total_loss/total_test_num, accuracy/total_test_num))
    score = accuracy_score(y_true, y_pred)
    print(score)
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_true, y_pred)
    print(confusion_matrix)
    from sklearn.metrics import classification_report
    target_names = ['差评', '好评']
    print(classification_report(y_true, y_pred, target_names=target_names))

def train_model(train_iter, dev_iter, model, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    epochs = 10
    print('training...')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        accuracy = 0
        total_train_num = len(train_iter.dataset)
        for batch in train_iter:
            feature = batch.text
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            accuracy += (torch.argmax(logit, dim=1)==target).sum().item()
        print('>>> Epoch_{}, Train loss is {}, Accuracy:{} \n'.format(epoch,loss.item()/total_train_num, accuracy/total_train_num))
        model.eval()
        total_loss = 0.0
        accuracy = 0
        total_valid_num = len(dev_iter.dataset)
        for batch in dev_iter:
            feature = batch.text  # (W,N) (N)
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            out = model(feature)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            accuracy += (torch.argmax(out, dim=1)==target).sum().item()
        print('>>> Epoch_{}, Valid loss:{}, Accuracy:{} \n'.format(epoch, total_loss/total_valid_num, accuracy/total_valid_num))

def saveModel(model,name):
    torch.save(model, 'done_model/'+name+'_model.pkl')

model = TextRNN_Attention()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, val_iter, test_iter = DataSet.getIter()

if __name__ == '__main__':
    train_model(train_iter, val_iter, model, device)
    saveModel(model,'TextRNN_Attention')
    test_model(test_iter, model, device)

