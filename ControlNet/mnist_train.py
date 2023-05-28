from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from mnist_control_dataset import MNISTControlDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import datetime
import os


from functools import partial
from ldm.util import instantiate_from_config
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from omegaconf import OmegaConf

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = 0  # num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,  # if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = False
        init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)
# Configs
resume_path = './models/control_mnist_m6e30_take2.ckpt'
logger_freq = 500
learning_rate = 5e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
config_path = './models/model6_epoch30_control.yaml'
model = create_model(config_path).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


config = OmegaConf.load(config_path)
data = instantiate_from_config(config.data)
batch_size = data.batch_size
# NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
# calling these ourselves should not be necessary but it is.
# lightning still takes care of proper multiprocessing though
data.prepare_data()
data.setup()
# Initialize wandb
wandb.init(project="final-year-project", config={
    "learning_rate": learning_rate,
    "batch_size": batch_size,
})

name = 'control_mnist_m6e30_take2'
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
nowname = now + '_' + name
logdir = os.path.join('logs', nowname)
wandb_logger = WandbLogger(save_dir=logdir, log_model=True, name=nowname, id=name)

ckptdir = os.path.join(logdir, "checkpoints")
# Create a checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val/loss_simple_ema",  # the validation metric to monitor, change it as per your requirements
    dirpath=ckptdir,  # the directory where checkpoints will be saved
    filename="{epoch}-{step}-{val_loss:.6f}",  # a unique name for each checkpoint
    save_top_k=10,  # keep only the top 5 models
    mode="min",  # the goal is to minimize the monitored quantity
    save_last=True,
    verbose=True,
)

# Misc
# dataset = MNISTControlDataset()
# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback], logger=wandb_logger)


# Train!
trainer.fit(model, data)
