import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from model import EnhancedCNN

def train_model():
    # Data loader with augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model, loss, optimizer
    model = EnhancedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # Training loop
    model.train()
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Training Accuracy: {accuracy:.2f}%')

    return model, accuracy

if __name__ == "__main__":
    model, acc = train_model()
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
