import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import MNISTCNN


BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "C:\projects\microservice_ml\ml_service\data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Загрузка MNIST из локальной папки", DATA_ROOT)
train_dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Данные загружены: {len(train_dataset)} тренировочных, {len(test_dataset)} тестовых")

model = MNISTCNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

def train_epoch(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
    return total_loss / len(train_loader)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return accuracy

print("Начинаем обучение...")
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(epoch)
    print(f"End of epoch {epoch}: average loss = {loss:.4f}")
    test()

os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/mnist_cnn.pth")
print("Модель сохранена в models/mnist_cnn.pth")