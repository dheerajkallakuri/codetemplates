import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (784 nodes) to hidden layer (128 nodes)
        self.fc2 = nn.Linear(128, 64)       # Hidden layer (128 nodes) to another hidden layer (64 nodes)
        self.fc3 = nn.Linear(64, 10)        # Hidden layer (64 nodes) to output layer (10 nodes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)             # Flatten the input tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the neural network
model = SimpleNN()

criterion = nn.CrossEntropyLoss()           # Cross-entropy loss for classification
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Stochastic Gradient Descent optimizer

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

for epoch in range(10):  # number of epochs
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()   # zero the parameter gradients

        outputs = model(images) # forward pass
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()         # backward pass
        optimizer.step()        # update weights

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
