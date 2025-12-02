import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
    """
    Base class for CNN feature extractors.
    """
    def __init__(self, input_shape, output_dim=512):
        super(BaseCNN, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError

    def get_output_dim(self):
        return self.output_dim


class SmallCNN(BaseCNN):
    """
    A small LeNet-like CNN.
    Suitable for simple environments or quick testing.
    """

    def __init__(self, input_shape, output_dim=256):
        super(SmallCNN, self).__init__(input_shape, output_dim)
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        # Calculate flat size
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 5, 2), 5, 2), 3, 2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 5, 2), 5, 2), 3, 2)
        linear_input_size = convw * convh * 64

        self.fc = nn.Linear(linear_input_size, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class LargeCNN(BaseCNN):
    """
    A larger CNN with more convolutional layers.
    Suitable for more complex visual environments.
    """

    def __init__(self, input_shape, output_dim=512):
        super(LargeCNN, self).__init__(input_shape, output_dim)
        c, h, w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Extra layer
            nn.ReLU(),
        )

        # Calculate output size of features
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            dummy_output = self.features(dummy_input)
            self.flat_size = dummy_output.view(1, -1).size(1)

        self.fc = nn.Linear(self.flat_size, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class CustomCNN(BaseCNN):
    """
    Customizable CNN based on layer definitions.
    """

    def __init__(self, input_shape, layers_config, output_dim=512):
        """
        layers_config: List of dicts, e.g. [{'out_channels': 32, 'kernel_size': 8, 'stride': 4}, ...]
        """
        super(CustomCNN, self).__init__(input_shape, output_dim)
        c, h, w = input_shape

        layers = []
        in_channels = c
        for layer_cfg in layers_config:
            out_channels = layer_cfg["out_channels"]
            kernel_size = layer_cfg["kernel_size"]
            stride = layer_cfg.get("stride", 1)
            padding = layer_cfg.get("padding", 0)

            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            dummy_output = self.features(dummy_input)
            self.flat_size = dummy_output.view(1, -1).size(1)

        self.fc = nn.Linear(self.flat_size, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class VGGCNN(BaseCNN):
    """
    A CNN architecture inspired by VGG-11/13.
    Uses blocks of Conv2d + ReLU followed by MaxPool2d.
    Features a heavy classifier head with fully connected layers.
    """
    def __init__(self, input_shape, output_dim=512):
        super(VGGCNN, self).__init__(input_shape, output_dim)
        c, h, w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            dummy_output = self.features(dummy_input)
            self.flat_size = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCNN(BaseCNN):
    """
    A ResNet-18 inspired architecture.
    Uses Residual Blocks (BasicBlock) to allow training deeper networks.
    """
    def __init__(self, input_shape, output_dim=512):
        super(ResNetCNN, self).__init__(input_shape, output_dim)
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


def get_model(model_name, input_shape, output_dim=512, **kwargs):
    if model_name == "small":
        return SmallCNN(input_shape, output_dim)
    elif model_name == "large":
        return LargeCNN(input_shape, output_dim)
    elif model_name == "custom":
        return CustomCNN(input_shape, kwargs.get("layers_config"), output_dim)
    elif model_name == "resnet":
        return ResNetCNN(input_shape, output_dim)
    elif model_name == "vgg":
        return VGGCNN(input_shape, output_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
