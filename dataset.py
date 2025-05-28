import torch
import torchvision
from torchvision.transforms.v2 import PILToTensor, Compose
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


class MNIST(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(root='./data', train=is_train, download=True)
        self.convert = Compose([PILToTensor()])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.convert(image)
        return image / 255.0, label
    
if __name__ == '__main__':
    dataset = MNIST()
    img, label = dataset[6]
    print(label)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()