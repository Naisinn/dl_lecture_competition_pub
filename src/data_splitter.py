import os
import torch

# データセットをチャンクに分割する関数
def split_dataset_into_chunks(data_path, chunk_size):
    print(f"----- Splitting: {data_path} -----")
    try:
        data = torch.load(data_path)
        print(f"Data loaded successfully. Shape: {data.shape}, Size: {data.numel()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    num_chunks = (len(data) + chunk_size - 1) // chunk_size
    print(f"Number of chunks: {num_chunks}")

    # チャンクファイルを保存するディレクトリを作成
    chunk_dir = os.path.splitext(data_path)[0] + "_chunks"
    print(f"Chunk directory: {chunk_dir}")
    os.makedirs(chunk_dir, exist_ok=True)

    for i in range(num_chunks):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        chunk_path = os.path.join(chunk_dir, f"chunk_{i}.pt")
        print(f"Saving chunk {i} to {chunk_path}...")
        try:
            torch.save(chunk, chunk_path)
            print(f"Chunk {i} saved successfully. Size: {chunk.numel()}")
        except Exception as e:
            print(f"Error saving chunk {i}: {e}")

if __name__ == "__main__":
    data_dir = "data"  # データディレクトリ
    chunk_size = 1000  # チャンクサイズ

    for split in ["train", "val", "test"]:
        split_dataset_into_chunks(os.path.join(data_dir, f"{split}_X.pt"), chunk_size)
        split_dataset_into_chunks(os.path.join(data_dir, f"{split}_subject_idxs.pt"), chunk_size)
        if split in ["train", "val"]:
            split_dataset_into_chunks(os.path.join(data_dir, f"{split}_y.pt"), chunk_size)