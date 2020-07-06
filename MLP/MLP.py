import numpy as np
import torch as pt
import torchvision as ptv

# import data
train_set = ptv.datasets.MNIST(
    "F:\Rutgers\\3rdSemester\Special Problem\MLP\dataset\\train",
    train=True,
    transform=ptv.transforms.ToTensor(),
    download=True)
test_set = ptv.datasets.MNIST(
    "F:\Rutgers\\3rdSemester\Special Problem\MLP\dataset\\test",
    train=False,
    transform=ptv.transforms.ToTensor(),
    download=True)

train_dataset = pt.utils.data.DataLoader(train_set, batch_size=100)
test_dataset = pt.utils.data.DataLoader(test_set, batch_size=100)


# network construct
class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = pt.nn.Linear(784, 512)
        self.fc2 = pt.nn.Linear(512, 128)
        self.fc3 = pt.nn.Linear(128, 10)

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout_in = pt.nn.functional.relu(self.fc1(din))
        dout_hi = pt.nn.functional.relu(self.fc2(dout_in))
        dout = pt.nn.functional.softmax(self.fc3(dout_hi), dim=1)

        return dout


model = MLP().cuda()
print(model)

# loss fucntion, optimizer & accuracy
optimizer = pt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_func = pt.nn.CrossEntropyLoss().cuda()


def AccuracyCompute(pre, label):
    pre = pre.cpu().data.numpy()
    label = label.cpu().data.numpy()

    test_np = (np.argmax(pre, 1) == label)
    test_np = np.float32(test_np)

    return np.mean(test_np)


# training process
def train():
    print("start training")
    for x in range(4):
        for i, data in enumerate(train_dataset):

            optimizer.zero_grad()

            (inputs, labels) = data
            inputs = pt.autograd.Variable(inputs).cuda()
            labels = pt.autograd.Variable(labels).cuda()

            outputs = model(inputs)

            loss = loss_func(outputs, labels)
            loss_val = loss
            loss.backward()

            optimizer.step()

            if i % 100 == 0 and i != 0:
                print("epoch ", i, "accracy:",
                      AccuracyCompute(outputs, labels), "loss:",
                      loss_val.item())


train()