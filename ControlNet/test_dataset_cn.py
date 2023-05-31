from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from mnist_control_dataset import MNISTControlDataset
from einops import rearrange
from PIL import Image
from torchvision.utils import make_grid

from tqdm import tqdm
import os

# model = create_model('./models/model12_epoch152_control.yaml').cpu()
# model.load_state_dict(load_state_dict('./logs/2023-05-30T22-50-32_control_mnist_m12e152/checkpoints/epoch=78-step=177749-val_loss=0.000000.ckpt', location='cuda'))
model = create_model('./models/model15_epoch181_control.yaml').cpu()
model.load_state_dict(load_state_dict('./logs/2023-05-30T22-53-05_control_mnist_m15e181/checkpoints/epoch=45-step=103499-val_loss=0.000000.ckpt', location='cuda'))

model = model.cuda()
ddim_sampler = DDIMSampler(model)

# outpath = './cn_test_outputs/m12e152_3'
outpath = './cn_test_outputs/m15e181_2'

os.makedirs(outpath, exist_ok=True)

prompt_file = "./test_prompts_dup.txt"
# seed = 42
seed = 79637
# seed = 92923
seed_everything(seed)
C = 4
H = 64
W = 64
f = 4
scale = 7.5
num_samples = 8
guess_mode = False
ddim_steps = 50
strength = 1
eta = 0

print(f"reading prompts from {prompt_file}")
with open(prompt_file, "r") as file:
    data = file.read().splitlines()
    data_idx_prompt = [tuple(i.split("_")) for i in data]

def process():
    with torch.no_grad():
        for prompts in tqdm(data_idx_prompt, desc="data_idx_prompt"):
            idx, prompt = prompts

            sample_path = os.path.join(outpath, f"{idx}_{prompt}")
            os.makedirs(sample_path, exist_ok=True)
            base_count = len(os.listdir(sample_path))
            grid_count = len(os.listdir(outpath)) - 1

            uc = None
            if scale != 1.0:
                uc = model.get_learned_conditioning(num_samples * [""])
            if isinstance(prompt, tuple):
                prompt = list(prompt)
            n_prompt = num_samples * [prompt]
            c = model.get_learned_conditioning(n_prompt)
            shape = [C, H // f, W // f]

            control = MNISTControlDataset.convertLabelToHintTensor(prompt).cuda()
            control = torch.stack([control for _ in range(num_samples)], dim=0)


            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [c]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [uc]}
            # shape = (4, 16, 16) # (4, H // 8, W // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)
            x_samples = (((x_samples + 1.0) / 2.0) * 255.0).cpu().numpy().clip(0,255).astype(np.uint8)
            # x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            # x_samples = np.squeeze(x_samples, axis=1)
            results = [x_samples[i] for i in range(num_samples)]


            # Save image
            for x_sample in results:
                x_sample = rearrange(x_sample, 'c h w -> h w c')
                x_sample = x_sample.astype(np.uint8)
                x_sample = x_sample.squeeze(axis=-1)  # remove the singleton dimension
                # img = Image.fromarray(x_sample.astype(np.uint8))
                img = Image.fromarray(x_sample, 'L')  # 'L' mode is for grayscale images
                # img = put_watermark(img, wm_encoder)
                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                base_count += 1


            # Generate grid image
            grid = torch.stack([torch.tensor(results)], 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=num_samples)

            # to image
            grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
            img = Image.fromarray(grid.astype(np.uint8))
            # img = put_watermark(img, wm_encoder)
            img.save(os.path.join(sample_path, f'grid-{grid_count:04}.png'))
            grid_count += 1
        
    return results

if __name__ == '__main__':
    process()

    