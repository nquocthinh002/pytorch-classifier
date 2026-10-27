import torch
import torch.nn as nn
from torchsummary import summary

class ClassifierModel(nn.Module):
    def __init__(self, num_classes=10, input_shape=(3, 32, 32)):
        super().__init__()
        self.input_shape = input_shape
        self.cnt_fc = 3

        #Convolutional layers
        cnt_channels = [3, 8, 16, 32]
        cnt_num_layers = [2, 3, 1]
        self.cnt_block = len(cnt_num_layers)
        for iter in range(self.cnt_block):
            in_channles = cnt_channels[iter]
            out_channels = cnt_channels[iter + 1]
            num_layers = cnt_num_layers[iter]
            setattr(self, f'conv{iter+1}', self.make_block(in_channels=in_channles, out_channels=out_channels, num_layers=num_layers))

        # self.conv1 = self.make_block(in_channels=3, out_channels=8, num_layers=2)
        # self.conv2 = self.make_block(in_channels=8, out_channels=16, num_layers=3)
        # self.conv3 = self.make_block(in_channels=16, out_channels=32, num_layers=1)

        # Calculate the output size after convolutions
        in_features = self.get_conv_output_size()

        #Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=32),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.SiLU(),
            nn.Dropout(0.3)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU()
        )
        self.fc_final = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, x):
        for iter in range(self.cnt_block):
            x = getattr(self, f'conv{iter + 1}')(x)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.shape[0], -1) # Flatten for fully connected layers

        for iter in range(self.cnt_fc):
            x = getattr(self, f'fc{iter + 1}')(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = self.fc_final(x)
        return x

    @staticmethod # A static method does not receive an implicit first argument. // does not require 'self'
    def make_block(in_channels, out_channels, kernel_size=3, padding=1, num_layers=2):
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels # update for next layer
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def get_conv_output_size(self):
        with torch.no_grad():
            x = torch.empty(1, *self.input_shape) # Batch size 1 with input shape
            for iter in range(self.cnt_block):
                x = getattr(self, f'conv{iter+1}')(x)
            # x = self.conv1(x)
            # x = self.conv2(x)
            # x = self.conv3(x)
            return x.numel() # Total number of elements in the tensor

if __name__ == '__main__':
    NUM_CLASSES = 10
    INPUT_SHAPE = (3, 32, 32)

    model = ClassifierModel(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)
    print(model.input_shape)
    print(model)
    summary(model, input_size=INPUT_SHAPE)

    data = torch.rand(8, *INPUT_SHAPE)
    # print(data.shape)
    output = model(data)
    print(output.shape)
    # print(model.get_conv_output_size())
