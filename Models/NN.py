import torch.nn as nn
import torch



class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden, epoch, learning_rate):
        super(NeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = int((self.input_dim + self.output_dim) * 2 / 3)
        self.num_hidden = num_hidden

        self.epoch = epoch
        self.learning_rate = learning_rate

        self.model = nn.ModuleList()
        self.model.append(torch.nn.Linear(self.input_dim, self.hidden_dim))
        if self.num_hidden != 1:
            for i in range(self.num_hidden-1):
                self.model.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
        self.model.append(torch.nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i != len(self.model):
                x = torch.selu(self.model[i](x))
            else:
                x = self.model[i](x)
        return x

    def fit(self, x, y):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        for epoch in range(self.epoch):
            inputs = torch.autograd.Variable(torch.Tensor(x))
            targets = torch.autograd.Variable(torch.Tensor(y))
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, self.epoch, loss.item()))