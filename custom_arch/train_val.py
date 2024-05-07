import os
import time
import torch
from model import MLP, CnnClassifier
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Import this

import torch_xla.core.xla_model as xm

EPOCHS = 4
BATCH_SIZE = 32
MODEL_ARCHITECTURE = 'cnn'  # or 'mlp'

train_dataset = mnist.MNIST(root='./MNIST_DATA_train', train=True, download=True, transform=ToTensor())
val_dataset = mnist.MNIST(root='./MNIST_DATA_val', train=False, download=True, transform=ToTensor())

def main():
    device = 'xla'
    print(f"DEVICE={device}")

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/mnist_experiment')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP().to(device) if MODEL_ARCHITECTURE == 'mlp' else CnnClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.NLLLoss() if MODEL_ARCHITECTURE == 'mlp' else torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_train_loss = 0
        for batch_idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.view(train_x.size(0), -1) if MODEL_ARCHITECTURE == 'mlp' else train_x
            train_x = train_x.to(device)
            train_label = train_label.to(device)

            optimizer.zero_grad()
            output = model(train_x)
            loss = loss_fn(output, train_label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            xm.mark_step()
            

        epoch_duration = time.time() - start_time
        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Time/epoch_duration', epoch_duration, epoch)

        # Validation part
        model.eval()
        total_val_loss = 0
        correct = 0
        with torch.no_grad():
            for val_x, val_label in val_loader:
                val_x = val_x.view(val_x.size(0), -1) if MODEL_ARCHITECTURE == 'mlp' else val_x
                val_x = val_x.to(device)
                val_label = val_label.to(device)
                val_output = model(val_x)
                val_loss = loss_fn(val_output, val_label)
                total_val_loss += val_loss.item()
                predicted = torch.argmax(val_output, dim=1)
                correct += (predicted == val_label).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct / len(val_dataset)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

    writer.close()
    print('----------End Training ---------------')

if __name__ == '__main__':
    main()
