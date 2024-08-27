import torch
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from .visualize_kernels import plot_block_with_kernels

__all__ = [
    "EarlyStoppingScoreHistory",
    "KLD_Weight_CosineAnnealing",
    "EarlyStopping",
    "plot_all",
    "plot_grad_flow",
    "plot_reconstructions"
]

class EarlyStoppingScoreHistory:
    def __init__(self, memory_size=10):
        self.last_scores = []   
        self.memory_size = memory_size

    def __getitem__(self, idx):
        return self.last_scores[idx]
    
    def __setitem__(self, idx, value):
        self.last_scores[idx] = value

    def append(self, value):
        self.last_scores.append(value)
        self.last_scores = self.last_scores[-self.memory_size:]

    def __len__(self):
        return len(self.last_scores)
    
    def __iter__(self):
        return iter(self.last_scores)
    
    def __str__(self):
        return str(self.last_scores)
    
    def __repr__(self):
        return repr(self.last_scores)
    
    def __contains__(self, item):
        return item in self.last_scores
    
    def __add__(self, other):
        return self.last_scores + other
    
    def __radd__(self, other):
        return other + self.last_scores
    
    def __iadd__(self, other):
        self.last_scores += other

    def get_trend(self):
        diff = [self.last_scores[i] - self.last_scores[i-1] for i in range(1, len(self.last_scores))]
        return sum(diff)/len(diff)

class KLD_Weight_CosineAnnealing:
    def __init__(self, min_val, max_val, cycle_length, warmup: int = 50, wait_to_start: int = 50):
        self.start_weight = max_val
        self.end_weight = min_val
        self.nr_epochs = cycle_length
        self.current_epoch = 0
        self.warmup = warmup
        self.wait = wait_to_start

    def __call__(self):
        self.current_epoch += 1
        if self.current_epoch < self.wait:
            return 0.0
        elif self.current_epoch < self.warmup + self.wait:
            return ((self.current_epoch - self.wait)/self.warmup) * self.start_weight
        return self.start_weight + (self.end_weight - self.start_weight) * (1 + torch.cos(torch.tensor(self.current_epoch/self.nr_epochs) * 3.141592653589793))/2

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, memory_size=20, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.trace_func = trace_func
        self.last_scores = EarlyStoppingScoreHistory(memory_size=memory_size)

    def __call__(self, val_loss, model, path):
        score = -val_loss
        self.last_scores.append(score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and self.last_scores.get_trend() >= -self.delta:
                self.early_stop = True
            elif self.counter >= self.patience and self.last_scores.get_trend() < -self.delta:
                self.trace_func("Early Stopping prevented because of downward trend on error.")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, path = None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if path is None:
            path = self.path
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def plot_all(model, valid_pic, n_kernels):
    with torch.no_grad():
        recon_mu = model.encoder.encode(model.img2block(valid_pic)[:, None]).detach().cpu()

    x  = recon_mu[:, 0*n_kernels:1*n_kernels].flatten()
    y  = recon_mu[:, 1*n_kernels:2*n_kernels].flatten()
    nu = recon_mu[:, 2*n_kernels:3*n_kernels].flatten()

    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Reconstructed latent space - shared color")
    axs[0].scatter(x, y, c=nu, cmap="jet")
    axs[0].set_title("Mean")

    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Reconstructed latent space - shared color")
    axs[0].scatter(x, y, c=nu, cmap="jet")
    axs[0].set_title("Mean")

    # plt.scatter(x, y, c=np.exp(xs)+np.exp(ys), cmap="Reds")
    # plt.colorbar()
    plt.figure()
    plt.title("Reconstructed latent space")
    plt.hexbin(x, y, marginals=True)

    with torch.no_grad():
        recon = model.encoder(model.img2block(valid_pic)[:, None]).detach().cpu()

    x  = recon[:, 0*n_kernels:1*n_kernels].flatten()
    y  = recon[:, 1*n_kernels:2*n_kernels].flatten()
    nu = recon[:, 2*n_kernels:3*n_kernels].flatten()

    plt.figure()
    plt.title("Reconstructed latent space - after repameterization")
    plt.scatter(x, y, c=nu, cmap="jet")
    plt.colorbar()
    plt.show()

    plt.figure()
    # Plot seaborn histogram overlaid with KDE
    ax = sns.histplot(data=x, bins=20, stat='density', alpha= 1, kde=True,
                    edgecolor='white', linewidth=0.5,
                    line_kws=dict(color='black', alpha=0.5, linewidth=1.5, label='KDE'))
    ax.get_lines()[0].set_color('black') # edit line color due to bug in sns v 0.11.0
    # Edit legemd and add title
    ax.legend(frameon=False)
    ax.set_title('Seaborn histogram overlaid with KDE', fontsize=14, pad=15)
    plt.show()

    plt.figure()
    plt.title("Reconstructed latent space - after repameterization")
    plt.hexbin(x, y, nu, marginals=True)
    plt.show()

def plot_grad_flow(named_parameters):
    import matplotlib.pyplot as plt
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                print(f"None gradient for {n}")
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    fig = plt.figure()
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    return fig

def plot_reconstructions(model, image, device):
        orig_blocks = model.img2block(image, use_old=True)[:, None].to(device)
        encoded = model.encoder(orig_blocks)
        reconstructed_blocks = model.decoder(encoded)
        _recon = model.block2img(reconstructed_blocks, use_old=True)
        recon = _recon.detach().cpu().squeeze()
        ordered = ((reconstructed_blocks[:, None] - orig_blocks)**2).flatten(start_dim=1).mean(dim=1).sort()
        inds = ordered.indices
        best_recon = reconstructed_blocks[inds[0], :].squeeze().detach().cpu()
        worst_recon = reconstructed_blocks[inds[-1], :].squeeze().detach().cpu()
        median_recon = reconstructed_blocks[inds[len(inds)//2]].squeeze().detach().cpu()
        orig_best_recon = orig_blocks[inds[0], :].squeeze().detach().cpu()
        orig_worst_recon = orig_blocks[inds[-1], :].squeeze().detach().cpu()
        orig_median_recon = orig_blocks[inds[len(inds)//2], :].squeeze().detach().cpu()

        best_fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        best_fig.suptitle(f"Best Reconstruction\nMSE: {ordered.values[0]}")
        plt.sca(axs[0])
        plt.title("Reconstruction")
        plot_block_with_kernels(encoded[inds[0]], best_recon, block_size=model.block_size)
        plt.axis('off')
        plt.sca(axs[1])
        plt.title("Original")
        plt.imshow(orig_best_recon.transpose(0,1).detach().cpu(), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')

        median_fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        median_fig.suptitle(f"Median Reconstruction\nMSE: {ordered.values[len(inds)//2]}")
        plt.sca(axs[0])
        plt.title("Reconstruction")
        plot_block_with_kernels(encoded[inds[len(inds)//2]], median_recon, block_size=model.block_size)
        plt.axis('off')
        plt.sca(axs[1])
        plt.title("Original")
        plt.imshow(orig_median_recon.transpose(0,1).detach().cpu(), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        
        worst_fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        worst_fig.suptitle(f"Worst Reconstruction\nMSE: {ordered.values[-1]}")
        plt.sca(axs[0])
        plt.title("Reconstruction")
        plot_block_with_kernels(encoded[inds[-1]], worst_recon, block_size=model.block_size)
        plt.axis('off')
        plt.sca(axs[1])
        plt.title("Original")
        plt.imshow(orig_worst_recon.transpose(0,1).detach().cpu(), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        
        return recon, worst_fig, median_fig, best_fig