import sys
from keras.models import load_model

sys.path.append('./src')

from load_topos import load_topos
from models import DCEC

import wandb
from wandb.keras import WandbCallback
from time import gmtime, strftime

date = strftime("%Y-%m-%d %H:%M:%S", gmtime())
wandb.init(
    project='deep-artefact',
    group='001_train_dcec',
    name=f'001_train_dcec_{date}'
)


x = load_topos('./data/dummy.mat')
cae = load_model('./data/pretrain_cae_model.h5')

n_clusters = 61

dcec = DCEC(
    input_shape=x.shape[1:],
    CAE=cae,
    save_dir='./data',
    n_clusters=n_clusters,
    name=f'{n_clusters}'
)


history_dcec = dcec.fit(
    x.copy(),
    batch_size=256,
    maxiter=5,
    tol=1e-3,
    update_interval=1,
    callback=lambda delta: wandb.log({'delta': delta})
)
