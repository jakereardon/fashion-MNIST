import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim 
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

batch_size = 64

train_data = FashionMNIST(root='.', download=True, train=True, transform=ToTensor())
test_data = FashionMNIST(root='.', download=True, transform=ToTensor())
loader = DataLoader(train_data, batch_size)

class FashionMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 5)
        # 1x24x24
        self.conv2 = nn.Conv2d(16, 16, 4)
        # 1x21x21
        self.conv3 = nn.Conv2d(16, 16, 3)
        # 1x19x19
        self.lin = nn.Linear(19 * 19 * 16, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.lin(x)
        return x


classifier = FashionMNISTClassifier()
cost_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.005)

num_iters = 20

iter_costs = []
training_accuracy = []
for it in tqdm(range(num_iters)):
    running_cost = 0
    iter_accuracy = 0
    for sample in loader:
        optimizer.zero_grad()

        image, label = sample

        pred = classifier(image)

        iter_accuracy += sum(pred.argmax(1) == label) / batch_size

        cost = cost_func(pred, torch.squeeze(label))

        running_cost += cost

        cost.backward()
        optimizer.step()

    iter_costs.append(running_cost.item())
    training_accuracy.append(iter_accuracy / len(loader))

torch.save(classifier.state_dict(), "classifier.pt")

plt.plot(iter_costs)
plt.title("Cost")
plt.show()

plt.plot(training_accuracy)
plt.show()
