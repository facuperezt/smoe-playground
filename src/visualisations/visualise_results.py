#%%
import matplotlib.pyplot as plt
import torch

from src.data.dataloader import DataLoader
from src.models.elvira import Elvira2023Small, Elvira2023Full
from src.models.facu import ConvolutionalAutoencoder, VariationalAutoencoder, VqVae, ResNet

from src.utils import Img2Block, Block2Img

def get_class_name(instance) -> str:
    return repr(instance.__class__).strip("'>").split(".")[-1]

model = ResNet("base.json")
img2block = Img2Block(model.block_size, 384)
block2img = Block2Img(model.block_size, 384)
model.load_model("ResNetWeirdness_4_k_8_bs_real_ft_<latest>")
#%%
dataloader = DataLoader("dataset", data_path="professional_photos", img_size=384, batch_size=1)
# %%
img = dataloader.get_valid_pic()
# img = img.cuda()
plt.imshow(img.squeeze().cpu(), vmin=0, vmax=1, cmap="gray")
# %%
# model = model.cuda()
# %%
with torch.no_grad():
    out = model(img2block(img))

plt.imshow(block2img(out).detach().cpu().squeeze(), vmin=0, vmax=1, cmap="gray") 
#%%
import sys
import os
try:
    sys.path.insert(0, os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir))
except NameError:
    pass
from gmm_vis.src.main.GMMViz.GaussianMixtureModel import GMM
from gmm_vis.src.main.GMMViz.GmmPlot import GmmViz
import numpy as np

gmm = GMM(model.n_kernels)
gmm.data = np.empty((1, 2))
gmm.n_iter_ = 1
vis = GmmViz(gmm)

with torch.no_grad():
    smoe_params = model.encoder(img2block(img))

x = smoe_params[:, :model.n_kernels]
y = smoe_params[:, model.n_kernels : 2*model.n_kernels]
A = smoe_params[:, 3*model.n_kernels:].reshape([-1, model.n_kernels, 2, 2])
A = torch.tril(A)
sigma = torch.einsum('NKij,NKji -> NKij', A, A)
mean = torch.stack([x, y], axis=-1)*model.block_size
gmm.add_estimand_iteration(mean=np.asarray(mean)[0], sigma=np.asarray(sigma)[0])
vis.plot()
ax = plt.gca()
ax.imshow(out[0].squeeze())
# %%
