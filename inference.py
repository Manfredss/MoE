from dataset import MNIST
import torch
from moe import MNIST_MoE
from config import *
from torch.utils.data import DataLoader
import time

EPOCH = 10
BATCH_SIZE = 64
device = 'cpu'
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# device = 'cuda' if torch.cuda.is_available() else device

dataset = MNIST()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = MNIST_MoE(input_size=INPUT_SIZE, emb_size=EMB_SIZE, num_experts=EXPERT_NUM, top_k=TOP_K)
model.to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

start = time.time()
acc = 0
for epoch in range(EPOCH):
    for img, label in dataloader:
        logits = model(img.to(device))
        acc += (logits.cpu().argmax(-1)==label).sum()
acc = acc/len(dataset)*EPOCH
print(f'Using device: {device}, acc:{acc:.2f}%, time: {time.time()-start:.2f}s')