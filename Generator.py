import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, input_size, hidden_dimension, output_size):
        super().__init__()

        self.input_size = input_size

        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, hidden_dimension * 2)
        self.fc3 = nn.Linear(hidden_dimension * 2, hidden_dimension * 4)
        self.fc4 = nn.Linear(hidden_dimension * 4, output_size)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        if x.size(1) != self.input_size:
            x = x.view(-1, self.input_size)

        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))
        x = self.dropout(F.leaky_relu(self.fc3(x)))
        x = torch.tanh(self.fc4(x))

        return x