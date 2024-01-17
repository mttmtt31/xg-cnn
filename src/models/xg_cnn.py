import torch.nn as nn

# TODO: fix name of the script, return_indices=False

# Define the CNN model
class XGCNN(nn.Module):
    def __init__(self, dropout:float=0.0):
        super(XGCNN, self).__init__()
        # 2x30x40
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels = 2, out_channels = 16, kernel_size = (3, 3)),
            nn.BatchNorm2d(num_features = 16),
            nn.ReLU()
        ) # 16 x 28 x 38
        self.pooling_1 = nn.MaxPool2d(2,2, return_indices=False) # -> 16 x 14 x 19

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3)),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU()
        ) # -> 32 x 12 x 17
        self.pooling_2 = nn.MaxPool2d(2,2, return_indices=False) # -> 32 x 6 x 8

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3)),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU()
        ) # -> 64 x 4 x 6
        self.pooling_3 = nn.MaxPool2d(2,2, return_indices=False) # -> 64 x 2 x 3

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (1, 1)),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU()
        ) # -> 128 x 2 x 3

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 3, 64),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.pooling_1(x)
        x = self.conv_layer_2(x)
        x = self.pooling_2(x)
        x = self.conv_layer_3(x)
        x = self.pooling_3(x)
        x = self.conv_layer_4(x)
        x = self.fc(x)

        return x