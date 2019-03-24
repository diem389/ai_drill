# %%
import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True, dtype=torch.float)

b = a + 2
c = 2 * b * b
out = c.mean()
out.backward(retain_graph=True)

print(a.grad)

# %%
# Turn off grad operation
print(a.requires_grad)
with torch.no_grad():
    print((a**2).requires_grad)

# %%
b = a.detach()
print(b.requires_grad)

# %%
print(a.grad)
out.backward()
print(a.grad)

# %%
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size())
print(y.size())
print(z.size())

# %%
x = torch.randn(1)
print(x)
print(x.item())

# %%
# Converting a Torch Tensor to a NumPy array and vice versa is a breeze.
# The Torch Tensor and NumPy array will share their underlying memory locations,
# and changing one will change the other.

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)
# %%
b += 1
print(b)
print(a)

# %%
# cuda
x = torch.randn(4, 4)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

y = torch.ones_like(x, device=device)
print(x)
print(y)
# %%
x = x.to(device)
z = x + y
print(z)
print(z.to("cpu"))

# %%
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


net = Net()
print(net)

# %%
# The learnable parameters of a model are returned by net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size())

# %%
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# %%
# backward operation
net.zero_grad()
out.backward(torch.randn(1, 10))

# %%
# torch.nn only supports mini-batches. The entire torch.nn package only supports inputs
# that are a mini-batch of samples, and not a single sample.
# For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
# If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
output = net(input)
print(output)
target = torch.randn(10)
target = target.view(1, -1)

criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)
# %%
print(loss.item())

# %%
# Back propgation
net.zero_grad()

print('conv1.bais.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bais.grad after backward')
print(net.conv1.bias.grad)

# %% Update weights
# The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):
#      weight = weight - learning_rate * gradient

import torch .optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
loss = criterion(output, target)
loss.backward()
optimizer.step()  # does the update


# %%
# Loading images
import torchvision
import torchvision.transforms as transforms
