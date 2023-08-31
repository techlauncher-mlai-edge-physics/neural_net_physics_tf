'''
Bayesianised neural networks and training loop usage.
'''
from typing import Union

import torch
from torch.utils.data import DataLoader
from einops import rearrange
from src.nn_modules import FNO2d

from tqdm.autonotebook import tqdm



class FNO2d_BayesLL():
    def __init__(self, 
                pre_trained_nn: FNO2d, 
                torch_device: torch.device=torch.device('cpu'), 
                y_lik_var: float=1e1,
                w_prior_var: float=1e-4):
        # note: at this stage, pre_trained_nn must be an FNO2d. In the future, this class requirement may be relaxed
        self.device = torch_device
        pre_trained_nn = pre_trained_nn.to(device=torch_device)

        self.nn = pre_trained_nn
        self.last_layer = pre_trained_nn.last_layer

        self.in_channels = pre_trained_nn.in_channels
        self.nearly_out_channels = pre_trained_nn.out_2.in_features
        self.out_channels = pre_trained_nn.out_channels

        # assume initial prior on weights is: w ~ N(0, w_prior_var * I)
        # independence assumption on the weights for each output channel
        # +1 to allow for a bias term
        self.w_means = [torch.zeros(self.nearly_out_channels+1, 1).to(self.device) for _ in range(self.out_channels)]
        self.w_precs = [torch.eye(self.nearly_out_channels+1).to(self.device) / w_prior_var for _ in range(self.out_channels)]
        
        # further assume likelihood y ~ N(Xw, y_lik_var * I)
        self.y_sig2_hyp = y_lik_var
        
        # by default, don't assume data has to be normalized
        self.in_stds = None
        self.out_stds = None

        self.samplers_valid = False
        self.samplers = []


    def set_in_std(self, arg: Union[list, torch.Tensor]):
        # if standardisation is required for the inputs, use this
        if isinstance(arg, list):
            if len(arg) != self.in_channels:
                raise ValueError(f"Length of list of standard deviations does not agree with input channels to neural network; expected {self.in_channels} channels.")
            self.in_stds = arg

        elif isinstance(arg, torch.Tensor):
            if arg.shape[-1] != self.in_channels:
                raise ValueError(f"Input tensor X has incorrect number of channels; expected {self.in_channels} channels.")
            self.in_stds = [arg[..., i].std().item() for i in range(self.in_channels)]

        else:
            raise NotImplementedError("Attempt to set standard deviations in an unknown manner.")


    def set_out_std(self, arg: Union[list, torch.Tensor]):
        # if standardisation is required for the outputs, use this
        if isinstance(arg, list):
            if len(arg) != self.out_channels:
                raise ValueError(f"Length of list of standard deviations does not agree with output channels to neural network; expected {self.out_channels} channels.")
            self.out_stds = arg

        elif isinstance(arg, torch.Tensor):
            if arg.shape[-1] != self.out_channels:
                raise ValueError(f"Output tensor Y has incorrect number of channels; expected {self.out_channels} channels.")
            self.out_stds = [arg[..., i].std().item() for i in range(self.out_channels)]

        else:
            raise NotImplementedError("Attempt to set standard deviations in an unknown manner.")


    def train(self, dl:DataLoader, verbose:bool=False):
        # silly way to train; relies on a PyTorch dataloader being available
        # at this stage, it is assumed that the user passed in a conformable dataloader
        if verbose:
            print("Verbose output. Will display weight min and max values for the mean every 10 iterations.")

        self.samplers_valid = False
        with torch.no_grad():
            for step, (X, y) in enumerate(tqdm(dl)):
                X = X.to(self.device)
                y = y.to(self.device)
                # normalize
                if self.in_stds:
                    for i in range(self.in_channels):
                        X[..., i] /= self.in_stds[i]
                if self.out_stds:
                    for i in range(self.out_channels):
                        y[..., i] /= self.out_stds[i]
                
                # get the predictors for the Bayesian model
                predictors = self.last_layer(X)
                # stack all the X's and add bias term
                X = rearrange(predictors, 'b x y c -> (b x y) c') # be careful - X is now the result of the last layer, not the original predictors
                X = torch.hstack((torch.ones(X.shape[0],1).to(self.device), X))
                XTX = X.T@X
                y = rearrange(y, 'b x y c -> (b x y) c')

                new_means = []
                new_precs = []
                for i in range(self.out_channels):
                    y_i = y[:, i:i+1]

                    w_mean = self.w_means[i]
                    w_prec = self.w_precs[i]
                    new_prec = XTX/self.y_sig2_hyp + w_prec
                    new_mean = torch.linalg.solve(new_prec, (X.T@y_i)/self.y_sig2_hyp + w_prec@w_mean)
                    new_precs.append(new_prec)
                    new_means.append(new_mean)

                if verbose and (step + 1) % 10 == 0:
                    output_str = f'Step={step+1}'
                    for i in range(self.out_channels):
                        output_str += f', channel{i}: ({torch.min(new_means[i]).item()}, {torch.max(new_means[i]).item()})'
                    print(output_str + '.')

                self.w_means = new_means
                self.w_precs = new_precs

    def coef_(self):
        return (self.w_means, self.w_precs)

    def predict_mean(self, X:torch.Tensor, unnormalize_outputs:bool=True):
        out_device = X.device
        X = torch.clone(X.to(self.device))

        with torch.no_grad():
            if self.in_stds:
                for i in range(self.in_channels):
                    X[..., i] /= self.in_stds[i]

            predictors = self.last_layer(X)
            predictors = rearrange(predictors, 'b x y c -> (b x y) c') 
            predictors = torch.hstack((torch.ones(predictors.shape[0],1).to(self.device), predictors))
            
            w_means_stacked = torch.cat(self.w_means, dim=1)
            Y = predictors @ w_means_stacked

        Y = rearrange(Y, '(b x y) c -> b x y c', x=X.shape[1], y=X.shape[2])
        if unnormalize_outputs and self.out_stds:
            for i in range(self.out_channels):
                Y[..., i] *= self.out_stds[i]
        return Y.to(out_device)

    def rsample(self, X:torch.Tensor, unnormalize_outputs:bool=True):
        # for each sample in the batch, this function generates a subsequent prediction
        out_device = X.device
        X = torch.clone(X.to(self.device))

        # create new pytorch random samplers if the current ones are invalid
        n_sample = X.shape[0]

        if not self.samplers_valid:
            self.samplers = [
                torch.distributions.MultivariateNormal(
                    loc=torch.flatten(self.w_means[i]),
                    precision_matrix=self.w_precs[i]
                ) for i in range(self.out_channels)]
            self.samplers_valid = True

        with torch.no_grad():
            if self.in_stds:
                for i in range(self.in_channels):
                    X[..., i] /= self.in_stds[i]
            predictors = self.last_layer(X)
            predictors = rearrange(predictors, 'b x y c -> (b x y) c') 
            predictors = torch.hstack((torch.ones(predictors.shape[0],1).to(self.device), predictors))

            predictors = rearrange(predictors, '(b x y) c -> b (x y) c', x=X.shape[1], y=X.shape[2]) 
            # print(predictors.shape)
            w_sims = [sampler.rsample(torch.Size([n_sample])) for sampler in self.samplers]
            w_sims = [rearrange(w_sim, 'b c -> b c 1') for w_sim in w_sims]
            # print(w_sims[0].shape)
            samples = [predictors @ w_sims for w_sims in w_sims]
            # print(samples[0].shape)
           
        samples = torch.cat(samples, dim=2)
        samples = rearrange(samples, 'b (x y) c -> b x y c', x=X.shape[1], y=X.shape[2])
        if unnormalize_outputs and self.out_stds:
            for i in range(self.out_channels):
                samples[..., i] *= self.out_stds[i]
        return samples.to(out_device)

