#%%
import matplotlib.pyplot as plt
import torch

from src.data.dataloader import DataLoader
from src.models.elvira import Elvira2023Small, Elvira2023Full
from src.models.facu import ConvolutionalAutoencoder, VariationalAutoencoder

from src.utils import Img2Block, Block2Img


model = VariationalAutoencoder("manual_simple_ae.json")
img2block = Img2Block(model.block_size, 384)
block2img = Block2Img(model.block_size, 384)
model.load_state_dict(torch.load("vae_real_data_finetuned.pth"))
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

plt.imshow(block2img(out[0]).detach().cpu().squeeze(), vmin=0, vmax=1, cmap="gray")
#%%
# %%
