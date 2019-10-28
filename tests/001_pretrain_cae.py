import sys
sys.path.append('./src')

from load_topos import load_topos
from models import CAE
from training import pretrain_cae
import numpy as np

import wandb
from wandb.keras import WandbCallback
from time import gmtime, strftime

date = strftime("%Y-%m-%d %H:%M:%S", gmtime())
wandb.init(
    project='deep-artefact',
    group='001_pretrain_cae',
    name=f'001_pretrain_cae_{date}'
)

x = load_topos('./data/dummy.mat')
cae = CAE(input_shape=x.shape[1:], filters=[33, 64, 128, 61])
history = pretrain_cae(
    x,
    cae,
    batch_size=256,
    epochs=5,
    optimizer='adam',
    save_dir='./data/',
    callbacks=[WandbCallback()]
)

loss = history.history['loss']
epochs = np.arange(1, len(loss)+1)

np.save('./data/cae_loss.npy', np.array([loss, epochs]))
