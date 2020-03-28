import torch 
import numpy as np
import math
from data import load_mnist, plot_images, save_images
from torch.autograd import Variable
import torch.nn.functional as F 
import matplotlib.pyplot as plt

M = 100

# You may use the script provided in A2 or dataloaders provided by framework
# Load MNIST and Set Up Data
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = np.round(train_images[0:10000])
train_labels = train_labels[0:10000]
test_images = np.round(test_images[0:10000])

# sampler from Diagonal Gaussian x~N(μ,σ^2 I) (hint: use reparameterization trick here)
def rgaussian(mu, sigma):
	'''
	mu: [d] floattensor of mu values

	Sigma: [d] floattensor of sigma values
	'''
	return mu + sigma * Variable(torch.randn(mu.size()))

# sampler from Bernoulli
def rbernoulli(p_tensor):
	'''
    p_list: [d] the tensor of all p values
	'''
	p_list = p_tensor.cpu().numpy()

	result = np.empty(p_list.shape)
	for i in range(p_list.shape[0]):
		result[i] = np.random.choice([0,1], p=[1-p_list[i], p_list[i]])
	return torch.IntTensor(result)

# log-pdf of x under Diagonal Gaussian N(x|μ,σ^2 I)
def logpdf_gaussian(x, mu, sigma):
	'''
	x: [d] tensor of the input 

	mu: [d] tensor of mu

	sigma: [d] tensor of sigma
	'''
	k = x.shape[0]
	sigmaM = torch.diag_embed(1/sigma)
	up = torch.matmul(torch.matmul(torch.unsqueeze(-0.5*(x-mu),dim=0), sigmaM), x-mu)
	down = torch.log(torch.sqrt(((2*math.pi) ** k) * torch.det(sigmaM)))
	return up - down

# log-pdf of x under Bernoulli 
def logpdf_ber(x, p_list):
	'''
	x: [d] tensor of the input

	p_list: [d] tensor of the values of p
	'''
	return torch.add(x * torch.log(p_list), (1-x) * torch.log(1-p_list))

# Set latent dimensionality=2 and number of hidden units=500.
Dz = 2
Dh = 500
Dd = 784

class VAE(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_latent):
        super(VAE, self).__init__()
        self.Wb3 = torch.nn.Linear(n_feature, n_hidden)
        self.Wb4 = torch.nn.Linear(n_hidden, n_latent)
        self.Wb5 = torch.nn.Linear(n_hidden, n_latent)
        self.Wb1 = torch.nn.Linear(n_latent, n_hidden)
        self.Wb2 = torch.nn.Linear(n_hidden, n_feature)

    # Define MLP for recognition model / "encoder"
    # Provides parameters for q(z|x)
    # Define sample from recognition model
    # Samples z ~ q(z|x)
    def encoderMLP(self, x):
    	'''
    	x: [Dd] the input image with shape 784
    	'''
    	h = torch.tanh(self.Wb3(x))
    	mu = self.Wb4(h) 
    	log_sigma_square = self.Wb5(h)
    	return mu,log_sigma_square

    # Define MLP for generative model / "decoder"
    # Provides parameters for distribution p(x|z)
    
    def decoderMLP(self, z):
        '''
        z: [Dz] the generated z from encoder
        '''
        h1 = torch.tanh(self.Wb1(z))
        y = torch.sigmoid(self.Wb2(h1))

        return y

    def forward(self, x):
    	'''
    	x: [Dd] the input image x
    	'''
    	mu, log_sigma_square = self.encoderMLP(x)
    	sigma = torch.exp(log_sigma_square / 2)
    	z = rgaussian(mu, sigma)
    	y = self.decoderMLP(z)
    	return y, z, mu, sigma 

def loss_func(x, y, z, mu, sigma):

	# log_q(z|x) logprobability of z under approximate posterior N(μ,σ^2)
    log_q_z_x = logpdf_gaussian(z, mu, sigma)

    # log_p_z(z) log probability of z under prior
    iso_mu = torch.zeros(Dz)
    iso_sigma = torch.ones(Dz)
    #iso_gauss = rgaussian(iso_mu, iso_sigma)
    log_p_z = logpdf_gaussian(z, iso_mu, iso_sigma)

    # log_p(x|z) - conditional probability of data given latents.
    log_p_x_z = torch.sum(logpdf_ber(x, y))

    return log_q_z_x - log_p_x_z - log_p_z

if __name__ == "__main__":
    # Load Saved Model Parameters (if you've already trained)
    VAE = VAE(Dd, Dh, Dz)
    
    # Set up ADAM optimizer
    optimizer = torch.optim.Adam(VAE.parameters())

    # Train for ~200 epochs (1 epoch = all minibatches in traindata)
    for epoch in range(20):
        print("epoch {0} started".format(epoch))

        # Monte Carlo Estimator of mean ELBO with Reparameterization over M minibatch samples.
        # This is the average ELBO over the minibatch
        # Unlike the paper, do not use the closed form KL between two gaussians,
        # Following eq (2), use the above quantities to estimate ELBO as discussed in lecture
        for i in range(int(train_images.shape[0]/M)):
            loss = 0
            for m in range(M):
                x_numpy = train_images[100*i+m]
                x_t = torch.from_numpy(x_numpy).float()
                prediction, z, mu, sigma = VAE(x_t)

                loss += loss_func(x_t, prediction, z, mu, sigma)

            loss = loss / M
            print(loss)
            optimizer.zero_grad()
            loss.backward()

            # Save Optimized Model Parameters
            optimizer.step()
    
    # ELBO on training set
    elbo_train = 0
    for train_index in range(train_images.shape[0]):
        x_numpyt = train_images[train_index]
        x_tt = torch.from_numpy(x_numpyt).float()
        predictiont, zt, mut, sigmat = VAE(x_tt)

        elbo_train += loss_func(x_tt, predictiont, zt, mut, sigmat)
    elbo_train = elbo_train / train_images.shape[0]
    print("The average ELBO for training set is {0}".format(-1 * elbo_train))
    
    # ELBO on test set
    elbo_test = 0
    for test_index in range(test_images.shape[0]):
        x_numpye = test_images[test_index]
        x_te = torch.from_numpy(x_numpye).float()
        predictione, ze, mue, sigmae = VAE(x_te)

        elbo_test += loss_func(x_te, predictione, ze, mue, sigmae)
    elbo_test = elbo_test / test_images.shape[0]
    print("The average ELBO for testing set is {0}".format(-1 * elbo_test))


