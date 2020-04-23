import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

dim_sample = 784
dim_hidden = 512
dim_latent = 20
dropout_prob = 0.5
batch_size = 100
train_set = datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
test_set = datasets.MNIST(root='./data', train=False, download=False, transform=transforms.ToTensor())

torch.set_default_tensor_type(torch.cuda.FloatTensor)
all_data = []
for i in range(60000):
    all_data.append(train_set[i][0])
    
for i in range(10000):
    all_data.append(test_set[i][0])

all_data = torch.cat(tuple(all_data), dim=0)

data_loader = torch.utils.data.DataLoader(dataset=all_data, batch_size=batch_size, shuffle=True)


class Vae(torch.nn.Module):
    def __init__(self, dim_sample, dim_hidden, dim_latent, dropout_prob):
        super(Vae, self).__init__()
        """
        @parem dim_sample: dimenision of sample space
        @parem dim_hidden: dimenision of hidden space
        @param dim_latent: dimenision of latent space
        @parem dropout_prob: dropout probability
        """
        self.dim_sample = dim_sample
        self.dim_hidden1 = dim_hidden

        self.dim_latent = dim_latent
        self.dropout_prob = dropout_prob
        ### encoder layers
        self.enc_layer1 = nn.Linear(dim_sample, dim_hidden)
        self.enc_layer2 = nn.Linear(dim_hidden, dim_hidden)
        self.enc_layer_mu = nn.Linear(dim_hidden, dim_latent)
        self.enc_layer_log_sd = nn.Linear(dim_hidden, dim_latent)
        ### decoder layers
        self.dec_layer1 = nn.Linear(dim_latent, dim_hidden)
        self.dec_layer2 = nn.Linear(dim_hidden, dim_hidden)
        self.dec_layer3 = nn.Linear(dim_hidden, dim_sample)
        ### dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        self.batchnorm = nn.BatchNorm1d(dim_hidden)


        #nn.init.xavier_uniform_(self.enc_layer1.weight)
        #nn.init.xavier_uniform_(self.enc_layer2.weight)
        #nn.init.xavier_uniform_(self.enc_layer_mu.weight)
        #nn.init.xavier_uniform_(self.enc_layer_log_sd.weight)
        #nn.init.xavier_uniform_(self.dec_layer1.weight)
        #nn.init.xavier_uniform_(self.dec_layer2.weight)
        #nn.init.xavier_uniform_(self.dec_layer3.weight)

    def encoder(self, x):
        """
        @param x : tensor of size (batch_size, dim_sample)
        @returns mu: tensor of size (batch_size, dim_latent)
        @returns log_sd: tensor of size (batch_size,  dim_latent)
        """
        h = F.relu(self.enc_layer1(x))
        h = F.relu(self.enc_layer2(h))
        mu = self.enc_layer_mu(h)
        log_sd = self.enc_layer_log_sd(h)
        return mu, log_sd

    def decoder(self, z):
        """
        @param z : tensor of size (batch_size, dim_latent)
        @returns x_hat : tensor of size (batch_size, dim_sample)
        """
        x_hat = F.relu(self.dec_layer1(z))
        x_hat = F.relu(self.dec_layer2(x_hat))
        x_hat = torch.sigmoid(self.dec_layer3(x_hat))
        return x_hat

    def generate_z(self, mu, log_sd) -> torch.tensor:
        """
        @param mu: tensor of size (batch_size, dim_latent)
        @param log_sd: tensor of size (batch_size,  dim_latent)
        @returns: tensor of size (batch_size, dim_latent)
        """
        (batch_size, dim_latent) = mu.size()
        epsilon = torch.randn(batch_size, dim_latent)
        z = mu + epsilon * torch.exp(log_sd)
        return z

    def forward(self, x):
        """
        @param x: tensor of size (batch_size, dim_sample)
        @returns x_hat: tensor of size (batch_size, dim_sample)
        @returns mu: tensor of size (batch_size, dim_latent)
        @returns log_sd: tensor of size (batch_size, dim_latent)
        """
        mu, log_sd = self.encoder(x)
        z = self.generate_z(mu, log_sd)
        x_hat = self.decoder(z)
        return x_hat, mu, log_sd


def loss_function(x_hat, x, mu, log_sd):
    """
    @param x : tensor of size (batch_size, dim_sample)
    @param x_hat: tensor of size (batch_size, dim_sample)
    @param mu: tensor of size (batch_size, dim_latent)
    @param log_sd: tensor of size (batch_size)
    @returns MSE: scalar
    @returns KL: scalar
    """
    ( _, dim_sample) = x.size()
    (batch_size, dim_latent) = mu.size()
    KL_loss = 0.5*torch.sum(torch.exp(2*log_sd)+mu*mu-1-2*log_sd)
    MSE_loss = F.mse_loss(x_hat, x, reduction='sum')
    return (MSE_loss + KL_loss)/batch_size


vae = Vae(dim_sample=dim_sample, dim_hidden=dim_hidden, dim_latent=dim_latent, dropout_prob=dropout_prob)


if torch.cuda.is_available():
    vae.cuda()

vae.cuda()

optimizer = optim.Adam(vae.parameters())


def train(epoch):
    vae.train()
    total_loss = 0
    for batch_idx, batch_data in enumerate(data_loader):
        x = batch_data.cuda()
        optimizer.zero_grad()
        x_hat, mu, log_sd = vae.forward(x.view([-1, dim_sample]))
        loss = loss_function(x_hat, x.view([-1, dim_sample]), mu, log_sd)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch,batch_idx*len(batch_data),len(data_loader.dataset),100.*batch_idx/len(data_loader),loss.item()))

    print('Average loss: {:.06f}'.format(total_loss/len(data_loader)))

for epoch in range(10):
    train(epoch)

with torch.no_grad():
    z = torch.randn(64, dim_latent).cuda()
    sample = vae.decoder(z).cuda()
    save_image(sample.view(64, 1, 28, 28), './samples/sample_' + '.png')
