import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from einops import rearrange, repeat

rescale = lambda x: (x + 1.) / 2.

class ImageFolderDataset(Dataset):
    """一個簡單的 Dataset，用於從資料夾載入圖片。"""
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): 包含圖片的資料夾路徑。
            transform (callable, optional): 要應用於圖片的轉換。
        """
        # 使用 glob 找到所有常見格式的圖片檔案
        img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
        self.file_paths = []
        for ext in img_extensions:
            self.file_paths.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
        
        self.file_paths = sorted(self.file_paths) # 排序以確保順序一致
        self.transform = transform
        
        if not self.file_paths:
            raise FileNotFoundError(f"在 '{folder_path}' 中找不到任何圖片檔案。請確認路徑是否正確。")

        print(f"在 '{folder_path}' 中找到 {len(self.file_paths)} 張圖片。")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        # 使用 Pillow 載入圖片，並轉換為 RGB 格式
        # .convert("RGB") 非常重要，可以避免灰階或有 alpha 通道的圖片出錯
        image = Image.open(img_path).convert("RGB")
        
        # 應用指定的轉換 (Resize, ToTensor)
        if self.transform:
            image = self.transform(image)
        
        return image


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

def custom_to_pil(x):
    x = x.detach().cpu()
    # print(f"Image max: {x.max()}, min: {x.min()}")
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

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step

# Command to run the script:
# python3 encode_image.py 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resume", type=str, default="models/ldm/ffhq256/model.ckpt", nargs="?", help="load from logdir or checkpoint in logdir")
    parser.add_argument("--config", type=str, default="models/ldm/ffhq256/config.yaml", help="Path to the model config file")
    parser.add_argument("--output_dir", type=str, default="./dataset/ffhq256_encode", help="Directory to save encoded latent tensors.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loading.")
    # parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint file")
    # parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    # parser.add_argument("--output_dir", type=str, required=True, help="Directory to save encoded images")
    args = parser.parse_args()

    configs = OmegaConf.load(args.config)
    # print(f"Loaded configs: {configs}")
    # exit(0)
    # model = instantiate_from_config(configs.model.params.first_stage_config)
    # model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"], strict=False)
    # model.eval()
    # model.cuda()
    ckpt = args.resume
    gpu = True
    eval_mode = True
    model, global_step = load_model(configs, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    # model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"], strict=False)
    model.eval()
    model.cuda()

    config = configs.model.params.first_stage_config

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # image_folder = "./dataset/ffhq256"  # 預設的 FFHQ 資料夾路徑

    # ffhq_dataset = dataset = ImageFolderDataset(folder_path=image_folder, transform=transform)
    
    # data_loader = DataLoader(
    #     ffhq_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.num_workers, pin_memory=True
    # )

    # output_dir = args.output_dir if hasattr(args, 'output_dir') else "./dataset/ffhq256_encode"

    # print(f"Starting encoding. Latents will be saved to '{output_dir}'")
    # os.makedirs(output_dir, exist_ok=True)

    # for i, batch in enumerate(tqdm(data_loader, desc="Encoding images from folder")):
    #     # 因為我們的 Dataset __getitem__ 只返回圖片，所以 batch 就是圖片 tensor
    #     images = batch.to("cuda")

    #     with torch.no_grad():
    #         encoded_latents = model.encode(images)

    #     # 遍歷批次中的每一個潛在向量並儲存
    #     for j in range(encoded_latents.shape[0]):
    #         single_latent = encoded_latents[j]
            
    #         # 計算全域的圖片索引
    #         image_index = i * args.batch_size + j
            
    #         # 將檔名格式化為 00000.pt, 00001.pt, ...
    #         filename = f"{image_index:05d}.pt"
    #         output_path = os.path.join(args.output_dir, filename)
            
    #         # 無損儲存
    #         torch.save(single_latent.cpu(), output_path)

    # print(f"\nFinished encoding {len(ffhq_dataset)} images.")
    # print(f"Latent tensors are saved in: {output_dir}")

    img = transform(Image.open("./00000.png").convert("RGB"))  
    # img = Image.open("/home/howard/LDM_Patch/data/inpainting_examples/bench2.png").convert("RGB")
    img = img.unsqueeze(0)  # Add batch dimension
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image min: {img.min()}, max: {img.max()}")
    print(f"Image size: {img.size()}")
    if len(img.shape) == 3:
        img = img[..., None]
    # img = rearrange(img, 'b h w c -> b c h w')
    img = img.to(memory_format=torch.contiguous_format).float()
    img = img.to("cuda")

    encode_img = model.encode_first_stage(img)
    if isinstance(encode_img, DiagonalGaussianDistribution):
        print("Encoded image is a DiagonalGaussianDistribution, extracting mean.")
    print(f"Encoded image shape: {encode_img.shape}")

    decode_img = model.decode_first_stage(encode_img)

    encode_img = encode_img.cpu().detach().numpy()
    encode_img = encode_img.squeeze(0)  # Remove batch dimension
    print("encode image max: {}, min: {}".format(torch.tensor(encode_img).max(), torch.tensor(encode_img).min()))
    print("decode image max: {}, min: {}".format(torch.tensor(decode_img).max(), torch.tensor(decode_img).min()))
    # encode_img = custom_to_pil(torch.tensor(encode_img))
    # encode_img = Image.fromarray(encode_img.astype(np.uint8))
    # encode_img.save("output_encode_ffhq_RGB.png")
    # print(f"Loaded model from {args.ckpt}")
    # print(f"Model: {model}")

    # retry_encode = transforms.ToTensor()(Image.open("./output_encode_ffhq.png"))
    # retry_encode = retry_encode.unsqueeze(0)  # Add batch dimension
    # retry_encode = retry_encode.to(memory_format=torch.contiguous_format).float()
    # retry_encode = retry_encode.to("cuda")
    
    # retry_decode = model.decode(retry_encode)
    # print(f"Retry encoded image shape: {retry_decode.shape}")

    # retry_decode = retry_decode.cpu().detach().numpy()
    # retry_decode = retry_decode.squeeze(0)  # Remove batch dimension
    # # print("Retry image max: {}, min: {}".format(torch.tensor(decode_img).max(), torch.tensor(decode_img).min()))
    # retry_decode = custom_to_pil(torch.tensor(retry_decode))
    # retry_decode.save("output_retry_decode_ffhq_RGB.png")

    decode_img = decode_img.cpu().detach().numpy()
    decode_img = decode_img.squeeze(0)  # Remove batch dimension
    decode_img = custom_to_pil(torch.tensor(decode_img))
    decode_img.save("test_decode_range.png")
