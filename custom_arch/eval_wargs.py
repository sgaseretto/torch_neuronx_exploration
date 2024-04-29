import os
import time
import torch
from model import MLP, CnnClassifier

from torchvision.datasets import mnist, FashionMNIST
from torch.utils.data import DataLoader
import argparse
from torchvision.transforms import ToTensor, Compose, Normalize

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='cnn', choices=['cnn', 'mlp'])
parser.add_argument('--dataset', type=str, default='fashionmnist', choices=['mnist', 'fashionmnist'])
parser.add_argument('--model_name', type=str, default='checkpoint')
args = parser.parse_args()

# Constants based on arguments
MODEL_ARCHITECTURE = args.arch
DATASET = args.dataset
MODEL_NAME = args.model_name

# Dataset handling
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
if DATASET == 'mnist':
    test_dataset = mnist.MNIST(root='./data', train=False, download=True, transform=transform)
else:
    test_dataset = FashionMNIST('./data', train=False, transform=transform, download=True)

def main():
    torch.manual_seed(0)
    device = 'cpu'  # As per requirement, evaluation only uses CPU

    # Model selection
    if MODEL_ARCHITECTURE == 'mlp':
        model = MLP().to(device)
    elif MODEL_ARCHITECTURE == 'cnn':
        model = CnnClassifier().to(device)

    # Load checkpoint
    checkpoint_path = f'checkpoints/{MODEL_NAME}.pt'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}. Please check the file name and try again.")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    test_loader = DataLoader(test_dataset, batch_size=32)

    # Evaluation loop
    print('----------Evaluating---------------')
    match_count = 0
    model.eval()
    start = time.time()
    for idx, (test_x, test_label) in enumerate(test_loader):
        if MODEL_ARCHITECTURE == 'mlp':
            test_x = test_x.view(test_x.size(0), -1)
        test_x = test_x.to(device)
        test_pred = model(test_x)
        pred_label = torch.argmax(test_pred, dim=1)
        match_count += (pred_label == test_label).sum().item()

        if idx == 1:  # Start timing after a warmup
            start = time.time()

    # Compute statistics
    total_samples = (idx + 1) * 32
    accuracy = match_count / total_samples
    throughput = total_samples / (time.time() - start)
    print("Test throughput (samples/sec): {:.2f}".format(throughput))
    print("Accuracy: {:.4f}".format(accuracy))
    if accuracy < 0.92: print("Accuracy did not meet the expected threshold of 92%")
    print('----------Done Evaluating---------------')

if __name__ == '__main__':
    main()
