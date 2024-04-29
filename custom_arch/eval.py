import os
import time
import torch
from model import MLP, CnnClassifier

from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# XLA imports
# import torch_xla.core.xla_model as xm

# MODEL_ARCHITECTURE = 'mlp'
MODEL_ARCHITECTURE = 'cnn'
DATASET = 'mnist'

# Load MNIST test dataset
test_dataset = mnist.MNIST(root='./MNIST_DATA_test', \
                           train=False, download=True, transform=ToTensor())

def main():
    # Fix the random number generator seeds for reproducibility
    torch.manual_seed(0)

    # XLA: Specify XLA device (defaults to a NeuronCore on Trn1 instance)
    # device = xm.xla_device()
    device = 'cpu'
    # Move model to device 
    if MODEL_ARCHITECTURE == 'mlp':
        model = MLP().to(device)
    elif MODEL_ARCHITECTURE == 'cnn':
        model = CnnClassifier().to(device)
    
    # Load check point
    checkpoint = torch.load('checkpoints/checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    # Prepare data loader
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Run the evaluation loop 
    print('----------Evaluating---------------')
    match_count = 0
    model.eval()
    start = time.time()
    for idx, (test_x, test_label) in enumerate(test_loader):
        if MODEL_ARCHITECTURE == 'mlp':
            # train_x = train_x.view(train_x.size(0), -1)
            test_x = test_x.view(test_x.size(0), -1)
        test_x = test_x.to(device)
        test_pred = model(test_x)
        pred_label = torch.argmax(test_pred, dim=1)
        match_count += sum(pred_label == test_label.to(device))
        # xm.mark_step() # XLA: collect ops and run them in XLA runtime
        if idx < 2: # skip warmup iterations
            start = time.time()
    
    # Compute statistics
    interval = idx - 2 # skip warmup iterations
    throughput = interval / (time.time() - start)
    print("Test throughput (iter/sec): {}".format(throughput))
    
    accuracy = match_count / (idx * 32)
    print("Accuracy: {}".format(accuracy))
    assert(accuracy > 0.92)
    print('----------Done Evaluating---------------')

if __name__ == '__main__':
    main()