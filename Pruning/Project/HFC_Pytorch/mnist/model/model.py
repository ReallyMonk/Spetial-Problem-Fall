import torch
import torch.nn as nn
import torch.nn.functional as F
class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

class Model(nn.Module):
    def __init__(self, i_c=1, n_c=10):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(i_c, 32, 5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)

        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)


        self.flatten = Expression(lambda tensor: tensor.view(tensor.shape[0], -1))
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, n_c)


    def forward(self, x_i, _eval=False):

        if _eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()
            
        x_o = self.conv1(x_i)
        x_o = torch.relu(x_o)
        x_o = self.pool1(x_o)

        x_o = self.conv2(x_o)
        x_o = torch.relu(x_o)
        x_o = self.pool2(x_o)

        x_o = self.flatten(x_o)

        x_o = torch.relu(self.fc1(x_o))

        self.train()

        return self.fc2(x_o)


CHANNELS = 1
WIDTH = 128
HEIGHT = 128
LATENT = 128
NUM_LABELS = 10
HIDDEN = 10
GENERATOR_EPSILON = 1

class Classifier(nn.Module):
    def __init__(self, z_dim):
        super(Classifier, self).__init__()

        self.image_size = CHANNELS * WIDTH * HEIGHT
        self.conv1 = nn.Conv2d(CHANNELS, 8, 2, stride=1)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 32, 4, stride=2)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 4, stride=2)
        self.conv5_bn = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 6 * 6, z_dim)

    def encode(self, x):
        h = F.relu(self.conv1_bn(self.conv1(x)))
        h = F.relu(self.conv2_bn(self.conv2(h)))
        h = F.relu(self.conv3_bn(self.conv3(h)))
        h = F.relu(self.conv4_bn(self.conv4(h)))
        h = F.relu(self.conv5_bn(self.conv5(h)))
        h = h.view(-1, 256 * 6 * 6)
        return torch.nn.Softmax()(self.fc1(h))

    def forward(self, x, encode=True, mean=False):
        return self.encode(x)

if __name__ == '__main__':
    # i = torch.FloatTensor(4, 1, 28, 28)

    # n = Model()

    # print(n(i).size())
    net = Classifier(10)
    for name, data in net.state_dict().items():
        print(name, data.shape)

