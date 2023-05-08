#Write a class to test a trained autoencoder using image
# from the MNIST dataset
from ldm.util import instantiate_from_config

class AE:

    def __init__(self, firstStageConfig):
        model = instantiate_from_config(firstStageConfig)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = self.disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def disabled_train(self, mode):
        return self

ae = AE({
    "target": "ldm.models.autoencoder.AutoencoderKL",
    "params": {
        "monitor": "val/rec_loss",
        "ckpt_path": "logs/2023-05-08T09-35-35_mnist_autoencoder/checkpoints/epoch=000004.ckpt",
        "embed_dim": 3,
        "lossconfig": {
            "target": "ldm.modules.losses.LPIPSWithDiscriminator",
            "params": {
                "disc_start": 50001,
                "kl_weight": 0.000001,
                "disc_weight": 0.5,
                "disc_in_channels": 1,
            },
        },
        "ddconfig": {
          "double_z": True,
          "z_channels": 1,
          "resolution": 256,
          "in_channels": 1,
          "out_ch": 1,
          "ch": 32,
          "ch_mult": [ 1,2,4 ],  # num_down = len(ch_mult)-1,
          "num_res_blocks": 2,
          "attn_resolutions": [ ],
          "dropout": 0.0,
        }
    }
})
