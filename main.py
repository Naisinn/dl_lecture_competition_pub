import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
import gc

from src.models import BasicConvClassifier
from src.utils import set_seed

# データセットクラス
class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", chunk_size: int = 1000) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.chunk_size = chunk_size

        self.X_paths = [
            os.path.join(data_dir, f"{split}_X.pt.chunk_{i}")
            for i in range((len(self) + chunk_size - 1) // chunk_size)
        ]
        self.subject_idxs_paths = [
            os.path.join(data_dir, f"{split}_subject_idxs.pt.chunk_{i}")
            for i in range((len(self) + chunk_size - 1) // chunk_size)
        ]
        if split in ["train", "val"]:
            self.y_paths = [
                os.path.join(data_dir, f"{split}_y.pt.chunk_{i}")
                for i in range((len(self) + chunk_size - 1) // chunk_size)
            ]
        else:
            self.y_paths = None

    def __len__(self) -> int:
        # 全体のデータ数を返すように修正
        return len(self.X_paths) * torch.load(self.X_paths[0]).shape[0]  

    def __getitem__(self, i):
        chunk_idx = i // self.chunk_size
        offset = i % self.chunk_size

        X = torch.load(self.X_paths[chunk_idx])[offset]
        subject_idxs = torch.load(self.subject_idxs_paths[chunk_idx])[offset]

        if self.y_paths is not None:
            y = torch.load(self.y_paths[chunk_idx])[offset]
            return X, y, subject_idxs
        else:
            return X, subject_idxs

    @property
    def num_channels(self) -> int:
        return torch.load(self.X_paths[0]).shape[1]

    @property
    def seq_len(self) -> int:
        return torch.load(self.X_paths[0]).shape[2]

# カスタムサンプラー
class SubjectBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # subject_idxs はここではダミーとして扱う
        self.subject_idxs = np.array([0] * len(dataset))  
        self.unique_subjects = np.unique(self.subject_idxs)

    def __iter__(self):
        for i in range(0, len(self.unique_subjects), self.batch_size):
            batch_subjects = self.unique_subjects[i:i + self.batch_size]
            batch_indices = np.where(np.isin(self.subject_idxs, batch_subjects))[0]
            yield batch_indices

    def __len__(self):
        return len(self.unique_subjects) // self.batch_size

# チャンクサンプラー
class ChunkSampler(torch.utils.data.Sampler):
    def __init__(self, num_chunks, chunk_size):
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size

    def __iter__(self):
        for i in range(self.num_chunks):
            yield range(i * self.chunk_size, (i + 1) * self.chunk_size)

    def __len__(self):
        return self.num_chunks

# メイン関数
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    chunk_size = 1000 # 適切な値を設定

    train_set = ThingsMEGDataset("train", args.data_dir, chunk_size=chunk_size)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=ChunkSampler(len(train_set) // chunk_size, args.batch_size // 16),
        num_workers=4,  # 適切な値を設定
        pin_memory=True
    )

    val_set = ThingsMEGDataset("val", args.data_dir, chunk_size=chunk_size)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_sampler=ChunkSampler(len(val_set) // chunk_size, args.batch_size // 16),
        num_workers=2,  # 適切な値を設定
        pin_memory=True
    )

    test_set = ThingsMEGDataset("test", args.data_dir, chunk_size=chunk_size)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_sampler=ChunkSampler(len(test_set) // chunk_size, args.batch_size // 16),
        num_workers=2,  # 適切な値を設定
        pin_memory=True
    )

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Consider using a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)

    # 勾配蓄積のステップ数
    accumulation_steps = 16  # 必要に応じて調整

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for i, (X, y, subject_idxs) in enumerate(tqdm(train_loader, desc="Train")):
            # データをGPUに転送する前にfloat16に変換
            X = X.to(args.device).half()
            y = y.to(args.device)

            y_pred = model(X)
            loss = F.cross_entropy(y_pred, y)
            loss = loss / accumulation_steps  # 勾配蓄積のために損失をスケール

            train_loss.append(loss.item())

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

            # メモリ使用量を監視
            if i % 10 == 0:
                print(f"メモリ使用量: {torch.cuda.memory_summary()}")

            # 不要になったテンソルを削除
            del X, y, y_pred, loss
            gc.collect()
            torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
                X = X.to(args.device).half()
                y = y.to(args.device)

                y_pred = model(X)

                val_loss.append(F.cross_entropy(y_pred, y).item())
                val_acc.append(accuracy(y_pred, y).item())

                # 不要になったテンソルを削除
                del X, y, y_pred
                gc.collect()
                torch.cuda.empty_cache()

        # Update learning rate scheduler based on validation loss
        scheduler.step(np.mean(val_loss))

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = []
    model.eval()
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for X, subject_idxs in tqdm(test_loader, desc="Validation"):
            X = X.to(args.device).half()
            preds.append(model(X).detach().cpu())

            # 不要になったテンソルを削除
            del X
            gc.collect()
            torch.cuda.empty_cache()

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()