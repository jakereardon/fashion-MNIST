import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim 
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

train_data = FashionMNIST(root='.', download=True, train=True, transform=ToTensor())
test_data = FashionMNIST(root='.', download=True, transform=ToTensor())
loader = DataLoader(train_data)


class FashionMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(28 * 28, 28 * 28)
        self.lin2 = nn.Linear(28 * 28, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x


classifier = FashionMNISTClassifier()
cost_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.05)

num_iters = 10

iter_costs = []
iter_accs = []
for it in tqdm(range(num_iters)):
    running_cost = 0
    running_acc = 0
    for sample in loader:
        optimizer.zero_grad()

        image, label = sample

        pred = classifier(image)
        if torch.argmax(pred) == label:
            running_acc += 1

        cost = cost_func(pred, torch.squeeze(label))

        running_cost += cost

        cost.backward()
        optimizer.step()

    iter_costs.append(running_cost.item())
    iter_accs.append(running_acc / len(train_data))

plt.plot(iter_costs)
plt.title("Cost")
plt.show()

plt.title("Accuracy")
plt.plot(iter_accs)
plt.show()
