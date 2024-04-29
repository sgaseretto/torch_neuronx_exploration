import os
import time
import torch
from model import MLP, CnnClassifier
from torchvision.datasets import mnist, FashionMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='gpu', choices=['xla', 'gpu', 'cpu'])
parser.add_argument('--arch', type=str, default='cnn', choices=['cnn', 'mlp'])
parser.add_argument('--dataset', type=str, default='fashionmnist', choices=['mnist', 'fashionmnist'])
parser.add_argument('--model_name', type=str, default='checkpoint')
parser.add_argument('--epochs', type=int, default=4)
args = parser.parse_args()

# Global constants
EPOCHS = args.epochs
WARMUP_STEPS = 2
BATCH_SIZE = 32
MODEL_ARCHITECTURE = args.arch
PROCESSING_UNIT = args.device
DATASET = args.dataset

# Transformations
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Dataset selection
if DATASET == 'mnist':
    train_dataset = mnist.MNIST(root='./data', train=True, transform=transform, download=True)
else:
    train_dataset = FashionMNIST('./data', train=True, transform=transform, download=True)

def setup_device():
    device = 'cpu'
    if args.device == 'gpu':
        if torch.cuda.is_available():
            device = 'cuda'
            print("Using NVIDIA GPU")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print("Using Apple MPS")
        else:
            print("GPU and MPS not available, falling back to CPU")
    elif args.device == 'cpu':
        device = 'cpu'
        print("Using CPU")
    elif args.device == 'xla':
        try:
            # XLA imports
            import torch_xla.core.xla_model as xm
            # XLA imports for parallel loader and multi-processing
            import torch_xla.distributed.parallel_loader as pl
            import torch_xla.distributed.xla_multiprocessing as xmp
            from torch.utils.data.distributed import DistributedSampler
            device = xm.xla_device()
            print("Using XLA")
        except ImportError:
            print("XLA not available, falling back to CPU")
            device = 'cpu'
    # return torch.device(device)
    return device

def main(index, device_name):
    # device = setup_device()
    device = device_name
    if device == 'xla':
        world_size = xm.xrt_world_size()
    torch.manual_seed(0)

    # Prepare data loader
    train_device_loader = None
    if device == 'xla':
        train_sampler = None
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset,
                                            num_replicas=world_size,
                                            rank=xm.get_ordinal(),
                                            shuffle=True)
        train_loader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
                                sampler=train_sampler,
                                shuffle=False if train_sampler else True)
        # XLA MP: use MpDeviceLoader from torch_xla.distributed
        train_device_loader = pl.MpDeviceLoader(train_loader, device)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    model = MLP().to(device) if MODEL_ARCHITECTURE == 'mlp' else CnnClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.NLLLoss() if MODEL_ARCHITECTURE == 'mlp' else torch.nn.CrossEntropyLoss()

    # Training loop
    print(f'----------Training {MODEL_ARCHITECTURE} on {DATASET}---------------')
    model.train()
    for epoch in range(EPOCHS):
        start = time.time()
        epoch_start = time.time()
        selected_train_loader = train_device_loader if train_device_loader != None else train_loader
        # for idx, (train_x, train_label) in enumerate(train_loader):
        for idx, (train_x, train_label) in enumerate(selected_train_loader):
            optimizer.zero_grad()
            if MODEL_ARCHITECTURE == 'mlp':
                train_x = train_x.view(train_x.size(0), -1)
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            output = model(train_x)
            loss = loss_fn(output, train_label)
            loss.backward()
            if device == 'xla':
                xm.optimizer_step(optimizer) # XLA MP: performs grad allreduce and optimizer step
            else:
                optimizer.step()
            if idx < WARMUP_STEPS:
                start = time.time()
        print(f"Done EPOCH: {epoch} - LOSS: {loss.detach().to('cpu')}\t - TIME: {time.time() - epoch_start}")

    throughput = (len(train_loader) - WARMUP_STEPS) / (time.time() - start)
    print("Train throughput (iter/sec): {}".format(throughput))
    print("Final loss is {:0.4f}".format(loss.detach().to('cpu')))

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", f"{args.model_name}.pt")
    checkpoint = {'state_dict': model.state_dict()}
    if device == 'xla':
        xm.save(checkpoint,'checkpoints/checkpoint.pt')
    else:
        torch.save(checkpoint, checkpoint_path)
    print(f'----------End Training, model saved to {checkpoint_path} ---------------')

if __name__ == '__main__':
    device_name = setup_device()
    if device_name == 'xla':
         xmp.spawn(main, args=(device_name,))
    else:
        main(0, device_name)
