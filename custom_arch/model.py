import torch
import torch.nn as nn
import torch.nn.functional as F

# Declare 3-layer MLP for MNIST dataset
class MLP(nn.Module):
  def __init__(self, input_size = 28 * 28, output_size = 10, layers = [120, 84]):
      super(MLP, self).__init__()
      self.fc1 = nn.Linear(input_size, layers[0])
      self.fc2 = nn.Linear(layers[0], layers[1])
      self.fc3 = nn.Linear(layers[1], output_size)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return F.log_softmax(x, dim=1)
  
 # PyTorch models inherit from torch.nn.Module
# class CnnClassifier(nn.Module):
#     def __init__(self):
#         super(CnnClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class CnnClassifier(nn.Module):
    def __init__(self):
        super(CnnClassifier, self).__init__()
        # Sequentially define the convolution and pooling layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Sequentially define the fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # Apply feature extractors
        x = self.features(x)
        # Flatten the output from the feature extractors
        x = torch.flatten(x, 1)  # x.view(-1, 16 * 4 * 4) can be replaced by torch.flatten(x, 1)
        # Apply classifier
        x = self.classifier(x)
        return x