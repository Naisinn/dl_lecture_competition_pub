import os
import torch

# データセットをチャンクに分割する関数
def split_dataset_into_chunks(data_path, chunk_size):
    data = torch.load(data_path)
    num_chunks = (len(data) + chunk_size - 1) // chunk_size
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    for i in range(num_chunks):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        torch.save(chunk, f"{data_path}.chunk_{i}")

if __name__ == "__main__":
    data_dir = "data"  # データディレクトリ
    chunk_size = 1000  # チャンクサイズ

    for split in ["train", "val", "test"]:
        split_dataset_into_chunks(os.path.join(data_dir, f"{split}_X.pt"), chunk_size)
        split_dataset_into_chunks(os.path.join(data_dir, f"{split}_subject_idxs.pt"), chunk_size)
        if split in ["train", "val"]:
            split_dataset_into_chunks(os.path.join(data_dir, f"{split}_y.pt"), chunk_size)