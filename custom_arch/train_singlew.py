import os
import time
import torch
from model import MLP, CnnClassifier

from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# XLA imports
import torch_xla.core.xla_model as xm

# import torch_xla.distributed.parallel_loader as pl


# Global constants
EPOCHS = 4
WARMUP_STEPS = 2
BATCH_SIZE = 32
# MODEL_ARCHITECTURE = 'mlp'
MODEL_ARCHITECTURE = 'cnn'

# limit = 50

# Load MNIST train dataset
train_dataset = mnist.MNIST(root='./MNIST_DATA_train',
                            train=True, download=True, transform=ToTensor())

def main():
    # XLA: Specify XLA device (defaults to a NeuronCore on Trn1 instance)
    device = 'xla'
    print(f"DEVICE={device}")
    
    # Prepare data loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    # train_loader = pl.MpDeviceLoader(train_loader, device)

    # Fix the random number generator seeds for reproducibility
    torch.manual_seed(0)

    # Move model to device and declare optimizer and loss function
    model = MLP().to(device) if MODEL_ARCHITECTURE == 'mlp' else CnnClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.NLLLoss() if MODEL_ARCHITECTURE == 'mlp' else torch.nn.CrossEntropyLoss()

    # Run the training loop
    print(f'----------Training {MODEL_ARCHITECTURE} ---------------')
    model.train()
    for epoch in range(EPOCHS):
        print(f"epoch={epoch}")
        start = time.time()
        # print("start the training")
        for step, (train_x, train_label) in enumerate(train_loader):
            # print("inside the loop")
            optimizer.zero_grad()
            if MODEL_ARCHITECTURE == 'mlp':
                train_x = train_x.view(train_x.size(0), -1)
            # print("moving the elements to the device")
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            # print("doing the inference")
            output = model(train_x)
            loss = loss_fn(output, train_label)
            loss.backward()
            optimizer.step()
            xm.mark_step() # XLA: collect ops and run them in XLA runtime
            if step < WARMUP_STEPS: # skip warmup iterations
                start = time.time()
            
            print(f"{step} loss={loss.detach().to('cpu')}")
            # print(f"step={step}")
            # if limit == idx:
            #     break

    # Compute statistics for the last epoch
    interval = step - WARMUP_STEPS # skip warmup iterations
    throughput = interval / (time.time() - start)
    print("Train throughput (iter/sec): {}".format(throughput))
    
    # this fails
    # print("Final loss is {:0.4f}".format(loss.detach().to('cpu')))

    # Save checkpoint for evaluation
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {'state_dict': model.state_dict()}
    # XLA: use xm.save instead of torch.save to ensure states are moved back to cpu
    # This can prevent "XRT memory handle not found" at end of test.py execution
    print("saving the model")
    xm.save(checkpoint,'checkpoints/checkpoint.pt')

    print('----------End Training ---------------')

if __name__ == '__main__':
    # torch.set_default_tensor_type('torch.FloatTensor')
    main()
