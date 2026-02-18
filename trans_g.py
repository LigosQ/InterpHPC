import pandas as pd
import numpy as np
import torch,random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

## 加载数据
s_label = 'class' 
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

df_train = pd.read_csv("./train_samples.csv")
ytrain = np.array(df_train[s_label])
xtrain = np.array(df_train.drop(columns=[s_label]))
dftest = pd.read_csv("./test_samples.csv")
ytest = np.array(dftest[s_label])
xtest = np.array(dftest.drop(columns=[s_label])) 
print(xtrain.shape, xtest.shape)

## 构造数据集
class MyDataSet(Dataset): 
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).float()
        self.length = label.shape[0]
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return self.length

train_set = MyDataSet(xtrain,ytrain)
test_set = MyDataSet(xtest,ytest)

train_loader = DataLoader(train_set, batch_size = 5, shuffle=True)
test_loader = DataLoader(test_set, batch_size= len(test_set), shuffle=True)

x_train, y_train = next(iter(train_loader))
## 搭建Transformer网络
class CNN_Transformer_model(nn.Module):
    def __init__(self):
        super(CNN_Transformer_model, self).__init__()
        self.CNNlayer = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 4, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 7,stride = 3, padding = 0),
            nn.Conv1d(in_channels = 4, out_channels = 8, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 7,stride = 3, padding = 0),
            ) 
        self.Transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=8, nhead=4), 
                num_layers= 2,
            )
        self.fc2 = nn.Linear(1280, 4)
    
    def forward(self,x):
        x = self.CNNlayer(x) 
        x = x.permute(2, 0, 1) 
        x = self.Transformer_encoder(x) 
        x = x.permute(1, 2, 0) 
        x = torch.flatten(x, 1) 
        x = self.fc2(x) 
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = CNN_Transformer_model().to(device)
torch.save(model.state_dict(), 'cnn_transformer_model.pth')

optimizer = torch.optim.SGD(model.parameters(),lr = 0.001)
loss_func = nn.CrossEntropyLoss()

train_loss = 0
train_acc = 0
i_trainIter = 13
ls_acc_mIter = [] 
ls_train_y_pred = [] 
ls_first11_pred_30 = [] 
for epoch in range (i_trainIter):
    ls_acc_1iter = [] 
    ls_first11_pred = []
    for i,(x_train,y_train) in enumerate(train_loader):
        x_train = x_train.view(-1,1,1472) 
        x_train, y_train = x_train.to(device), y_train.to(device)
        output = model(x_train).to(device)
        loss = loss_func(output, y_train.long())
        _, y_pred = torch.max(output.data, dim=1)
        if epoch == i_trainIter - 1:
            ls_train_y_pred.extend(y_pred.cpu().numpy())
        if i <= 2:
            ls_first11_pred.extend(y_pred.cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = (y_pred == y_train).sum() / 5 
        ls_acc_1iter.append(train_acc.item()) 
        train_loss += loss.item()
        if (i+1) % 150 == 0: 
            print('[%d %5d] loss: %.3f acc: %.3f' % (epoch + 1, i + 1, train_loss / 10, train_acc))
            train_loss = 0.0
    avg_acc = np.mean(ls_acc_1iter)
    ls_acc_mIter.append(avg_acc) 
    ls_first11_pred_30.extend(ls_first11_pred) 
# 打印预测结果
print("Training completed.")

## 测试
correct = 0
total = 0
with torch.no_grad():
    for x_test,y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        x_test = x_test.view(-1,1,1472)
        outputs = model(x_test)
        _, y_pred = torch.max(outputs.data, dim=1)
        correct += (y_pred == y_test).sum()
print('Accuracy of the testing samples: %.3f %%' % (100 * correct / len(test_set)))