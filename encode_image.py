import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from einops import rearrange, repeat

rescale = lambda x: (x + 1.) / 2.

def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

def custom_to_pil(x):
    x = x.detach().cpu()
    print(f"Image max: {x.max()}, min: {x.min()}")
    # x = (x + 1.) / 2.
    # x = torch.clamp(x, -1., 1.)
    x = torch.clamp(x, 0., 1.)
    x = x.permute(1, 2, 0).numpy()
    # x = x.numpy()
    print(f"Image shape: {x.shape}")
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

# Command to run the script:
# python3 encode_image.py --config configs/latent-diffusion/ffhq-ldm-vq-4.yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the model config file")
    # parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint file")
    # parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    # parser.add_argument("--output_dir", type=str, required=True, help="Directory to save encoded images")
    args = parser.parse_args()

    configs = OmegaConf.load(args.config)
    # print(f"Loaded configs: {configs}")
    # exit(0)
    model = instantiate_from_config(configs.model.params.first_stage_config)
    # model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"], strict=False)
    model.eval()
    model.cuda()

    config = configs.model.params.first_stage_config

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = transform(Image.open("/home/howard/LDM_Patch/test_sample_diffusion/ffhq256/samples/00634478/2025-08-06-22-54-59/img/sample_-00001.png").convert("RGB"))
    # img = Image.open("/home/howard/LDM_Patch/data/inpainting_examples/bench2.png").convert("RGB")
    img = img.unsqueeze(0)  # Add batch dimension
    # print(f"Image shape: {img.shape}")
    # print(f"Image dtype: {img.dtype}")
    # print(f"Image min: {img.min()}, max: {img.max()}")
    # print(f"Image size: {img.size()}")
    # if len(img.shape) == 3:
    #     img = img[..., None]
    # img = rearrange(img, 'b h w c -> b c h w')
    img = img.to(memory_format=torch.contiguous_format).float()
    img = img.to("cuda")

    encode_img = model.encode(img)

    decode_img = model.decode(encode_img)

    # encode_img = encode_img.cpu().detach().numpy()
    # encode_img = encode_img.squeeze(0)  # Remove batch dimension
    # encode_img = custom_to_pil(torch.tensor(encode_img))
    # encode_img.save("output.jpg")
    # print(f"Loaded model from {args.ckpt}")
    # print(f"Model: {model}")
    decode_img = decode_img.cpu().detach().numpy()
    decode_img = decode_img.squeeze(0)  # Remove batch dimension
    decode_img = custom_to_pil(torch.tensor(decode_img))
    decode_img.save("output_decode_ffhq.jpg")
