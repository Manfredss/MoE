import torch
from dataset import MNIST
from moe import MNIST_MoE
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
from config import *

device = torch.device('cpu')
# device = torch.device('mps' if torch.mps.is_available() else 'cpu')
# device = 'cuda' if torch.cuda.is_available() else device

dataset = MNIST()
model = MNIST_MoE(input_size=INPUT_SIZE, emb_size=EMB_SIZE, num_experts=EXPERT_NUM, top_k=TOP_K)
model.to(device)

try:
    model.load_state_dict(torch.load('model.pth'))
except:
    pass

# setup training
EPOCH = 100
BATCH_SIZE = 64
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
iterNum = 0
for epoch in range(EPOCH):
    for img, label in dataloader:
        logits = model(img.to(device))
        loss = F.cross_entropy(logits, label.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iterNum % 1000 == 0:
            print(f'>> epoch: {epoch}, iter: {iterNum}, loss: {loss.item()}')
            torch.save(model.state_dict(), '.model.pth')
            os.replace('.model.pth', 'model.pth')
        iterNum += 1
    