import os
import time
import torch
from model import MLP, CnnClassifier

from torchvision.datasets import mnist, FashionMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from torchvision.transforms import ToTensor

# Global constants
EPOCHS = 4
WARMUP_STEPS = 2
BATCH_SIZE = 32
# MODEL_ARCHITECTURE = 'mlp'
MODEL_ARCHITECTURE = 'cnn'
PROCESSING_UNIT = 'xla'
XLA_AVAILABLE = False
# DATASET = 'mnist'
DATASET = 'fashionmnist'

try:
    if PROCESSING_UNIT == 'xla':
        # XLA imports
        import torch_xla.core.xla_model as xm

        # XLA imports for parallel loader and multi-processing
        import torch_xla.distributed.parallel_loader as pl
        from torch.utils.data.distributed import DistributedSampler

        # Initialize XLA process group for torchrun
        import torch_xla.distributed.xla_backend
        torch.distributed.init_process_group('xla')
        print('XLA Available and enabled')
        XLA_AVAILABLE = True
except ImportError:
    print("XLA Libraries not installed or supported. The program will continue without it.")
    XLA_AVAILABLE = False

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Load MNIST based train dataset
if DATASET == 'mnist':
    train_dataset = mnist.MNIST(root='./MNIST_DATA_train',
                                train=True, download=True, transform=transforms.ToTensor())
elif DATASET == 'fashionmnist':
    train_dataset = FashionMNIST('./data', train=True, transform=transform, download=True)
else:
    train_dataset = mnist.MNIST(root='./MNIST_DATA_train',
                                train=True, download=True, transform=transforms.ToTensor())

def main():
    # Prepare data loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    # Fix the random number generator seeds for reproducibility
    torch.manual_seed(0)

    # Move model to device and declare optimizer and loss function
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
            device = 'cpu'
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
            device = 'cpu'
    else:
        print("Apple Silicon available")
        mps_device = torch.device("mps")
        device = mps_device

    # device = 'cpu'
    if MODEL_ARCHITECTURE == 'mlp':
        model = MLP().to(device)
    elif MODEL_ARCHITECTURE == 'cnn':
        model = CnnClassifier().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    if MODEL_ARCHITECTURE == 'mlp':
        loss_fn = torch.nn.NLLLoss() # for mlp
    elif MODEL_ARCHITECTURE == 'cnn':
        loss_fn = torch.nn.CrossEntropyLoss() # for cnn

    # Run the training loop
    print(f'----------Training {MODEL_ARCHITECTURE} on {DATASET}---------------')
    model.train()
    for epoch in range(EPOCHS):
        start = time.time()
        epoch_start = time.time()
        for idx, (train_x, train_label) in enumerate(train_loader):
            optimizer.zero_grad()
            if MODEL_ARCHITECTURE == 'mlp':
                train_x = train_x.view(train_x.size(0), -1)
            train_x = train_x.to(device)      
            train_label = train_label.to(device)
            output = model(train_x)
            loss = loss_fn(output, train_label)
            loss.backward()
            optimizer.step()
            if idx < WARMUP_STEPS: # skip warmup iterations
                start = time.time()
        print(f"Done EPOCH: {epoch} - LOSS: {loss.detach().to('cpu')}\t - TIME: {time.time() - epoch_start}")

    # Compute statistics for the last epoch
    interval = idx - WARMUP_STEPS # skip warmup iterations
    throughput = interval / (time.time() - start)
    print("Train throughput (iter/sec): {}".format(throughput))
    print("Final loss is {:0.4f}".format(loss.detach().to('cpu')))

    # Save checkpoint for evaluation
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {'state_dict': model.state_dict()}
    torch.save(checkpoint,'checkpoints/checkpoint.pt')
    print('----------End Training ---------------')

if __name__ == '__main__':
    main()