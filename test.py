
"""# 导入python包"""

import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
eng.cd('./data')
test_number = 1000
test_data = eng.data1(test_number)
print(test_data)
test_data = list(test_data)

def per_data0(train_data):
    train_data = list(train_data)
    train = []
    for i in range(len(train_data)):
        train_data1 = train_data[i]
        train1 = []
        for c in train_data1:
            x = np.array(c)
            x1 = np.real(x)  # 实数
            x2 = np.imag(x)  # 虚数
            train1.append(x1)
            train1.append(x2)
        train.append(np.array(train1))
    train = np.array(train)
    return train

test = per_data0(test_data)

val_ds = torch.tensor(test).float()
bs = test_number

val_dl = DataLoader(val_ds, batch_size=bs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('use:',device)

"""# V1 构建模型 普通模型"""


class Aotuencoder(nn.Module):
    def __init__(self, ):
        super(Aotuencoder, self).__init__()
        self.encoder = nn.Sequential(
          nn.Linear(184, 160),
          nn.ReLU(),
          nn.Linear(160, 140),
          nn.ReLU(),
          nn.Linear(140, 120),
          nn.ReLU(),
        )
        self.decoder = nn.Sequential(
          nn.Linear(120, 140),
          nn.ReLU(),
          nn.Linear(140, 160),
          nn.ReLU(),
          nn.Linear(160, 552),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

loss_func = nn.MSELoss().to(device)

aotuencoder =Aotuencoder().to(device)
aotuencoder.load_state_dict(torch.load("./model/best_model.pth"))
enc = OneHotEncoder()
data = [[0, 1],
        [1, 1],
        [2, 1],
        [3, 0],
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 1]]
enc.fit(data)
with torch.no_grad():
    for i, data in enumerate(val_dl):
        xb = data
        xb = xb.to(device)
        pred = aotuencoder(xb)
        pred = pred.cpu().detach().numpy()
    m = []
    for j in range(len(pred)):
        c1 = pred[j]
        c2 = c1.reshape(92,6)
        for l in range(len(c2)):
            c3 = c2[l]
            c4 = enc.inverse_transform(c3.reshape(1, -1))
            m.append(c4)

ps = []
for k in range(len(m)):
     k1 = m[k].tolist()
     if k1[0][0]==0:
        p0=-3
     elif k1[0][0]==1:
        p0 =-1
     elif k1[0][0]==2:
        p0=1
     else:
        p0=3

     if k1[0][1]==0:
        p1 = -1
     else:
        p1 = 1
     ps.append(p0)
     ps.append(p1)
ps1 = np.array(ps)
ps2 = ps1.reshape(bs,-1)



pred_s = []
for j in range(len(ps2)):
    p11 = ps2[j]
    pp1 = []
    for k in range(0, p11.shape[0], 2):
        if p11[k + 1] < 0:
            p2 = "%f%fi" % (p11[k], p11[k + 1])
        else:
            p2 = "%f+%fi" % (p11[k], p11[k + 1])
        pp1.append(p2)
    pred_s.append(pp1)
pred_s = np.array(pred_s)
np.savetxt('./out/pred_s.txt', pred_s, fmt="%s")


