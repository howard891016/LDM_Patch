# analyze_latents.py
import torch
import glob
import os
from tqdm import tqdm

def analyze_latent_stats(latents_folder):
    """
    計算資料夾中所有 .pt 潛在向量的統計數據。
    """
    file_paths = glob.glob(os.path.join(latents_folder, "*.pt"))
    if not file_paths:
        print(f"在 '{latents_folder}' 中找不到任何 .pt 檔案。")
        return

    print(f"正在分析 {len(file_paths)} 個潛在向量檔案...")

    # 先載入所有 tensors，收集起來
    all_tensors = [torch.load(p) for p in tqdm(file_paths, desc="Loading tensors")]
    
    # 將 list of tensors 合併成一個巨大的 tensor 以進行高效計算
    full_dataset_tensor = torch.stack(all_tensors)
    
    # 計算統計數據
    mean_val = full_dataset_tensor.mean()
    std_val = full_dataset_tensor.std()
    min_val = full_dataset_tensor.min()
    max_val = full_dataset_tensor.max()

    print("\n--- 潛在空間統計分析結果 ---")
    print(f"整體平均值 (Mean):   {mean_val.item():.4f}")
    print(f"整體標準差 (Std Dev): {std_val.item():.4f}")
    print(f"最小值 (Min):       {min_val.item():.4f}")
    print(f"最大值 (Max):       {max_val.item():.4f}")
    print("---------------------------------")


if __name__ == "__main__":
    # 將此路徑改為您存放 .pt 檔案的資料夾
    LATENTS_DIR = "./dataset/ffhq256_encode"
    analyze_latent_stats(LATENTS_DIR)