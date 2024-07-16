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

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

# Define a custom sampler to load data for all subjects in a batch
class SubjectBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.subject_idxs = np.array(dataset.subject_idxs)
        self.unique_subjects = np.unique(self.subject_idxs)

    def __iter__(self):
        for i in range(0, len(self.unique_subjects), self.batch_size):
            batch_subjects = self.unique_subjects[i:i + self.batch_size]
            batch_indices = np.where(np.isin(self.subject_idxs, batch_subjects))[0]
            yield batch_indices

    def __len__(self):
        return len(self.unique_subjects) // self.batch_size

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    # Use pin_memory=True for faster data transfer to GPU
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=SubjectBatchSampler(train_set, args.batch_size // 16),
        num_workers=0,  # num_workers を 0 に設定
        pin_memory=True
    )

    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size // 16,
        shuffle=False,
        num_workers=0,  # num_workers を 0 に設定
        pin_memory=True
    )

    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size // 16,
        shuffle=False,
        num_workers=0,  # num_workers を 0 に設定
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
    accumulation_steps = 4

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
            if i % 10 == 0:  # 10バッチごとに表示
                print(f"メモリ使用量: {torch.cuda.memory_summary()}")

            # 不要になったテンソルを削除
            del X, y, y_pred, loss
            gc.collect()
            torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
                X = X.to(args.device).half()  # データをGPUに転送する前にfloat16に変換
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
            X = X.to(args.device).half()  # データをGPUに転送する前にfloat16に変換
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