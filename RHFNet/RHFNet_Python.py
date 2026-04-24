"""RHFNet Image Classification Script Python Version
Former Author: JoeyBG.
Improved By: JoeyBG.
Date: 2026-04-07.
Affiliate: Beijing Institute of Technology.
Platform: Python with emdm environment.

Introduction:
  This script implements the RHFNet based on FasterNet-T0 architecture
      for radar human activity recognition image classification.
  It implements the Partial Convolution module in PyTorch to reduce
      redundant computations and memory access.
  The dataset is loaded from a folder tree and split randomly into an
      8:2 ratio for training and validation.
  No additional data augmentation is applied to the image data.
  The script trains the model using Adam optimizer and tracks the process.
  Finally, it exports customized visualizations for accuracy, loss curves,
      and the validation confusion matrix using the predefined style.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

print("---------- © Author: JoeyBG © ----------")

@dataclass
class RHFNetConfig:
    # Define dataset path and basic training parameters
    Dataset_Path: str = "SimH_RTM_Dataset"
    Num_Classes: int = 12                                                        # Number of activity labels in total
    Batch_Size: int = 32                                                         # Training batch size of the network
    Max_Epochs: int = 20                                                         # Maximum training epochs
    Initial_LR: float = 0.00147                                                  # Initial learning rate
    Train_Ratio: float = 0.8                                                     # Training split ratio for each class
    Image_Size: int = 224                                                        # Input image size of the network
    Num_Workers: int = 0                                                         # Safe worker count for Windows and Linux
    Random_Seed: int = 42                                                        # Random seed for reproducible splitting
    Torch_Num_Threads: int = 0                                                   # Positive value limits CPU training threads
    Output_Dir: str = "RHFNet_Results"                                           # Output folder for checkpoints and plots

    # Define the visualization style parameters
    Font_Name: str = "Palatino Linotype"
    Font_Size_Basis: int = 17
    Font_Size_Axis: int = 20
    Font_Size_Title: int = 22
    Font_Weight_Basis: str = "normal"
    Font_Weight_Axis: str = "normal"
    Font_Weight_Title: str = "bold"

    # Define the optimizer and checkpoint rule
    Optimizer_Name: str = "adam"
    Weight_Decay: float = 0.0
    Save_Best_By: str = "val_loss"

# Define the custom colormap colors for the visualization plots
JoeyBG_Colormap = np.array([
    [0.6196, 0.0039, 0.2588],
    [0.8353, 0.2431, 0.3098],
    [0.9569, 0.4275, 0.2627],
    [0.9922, 0.6824, 0.3804],
    [0.9961, 0.8784, 0.5451],
    [1.0000, 1.0000, 0.7490],
    [0.9020, 0.9608, 0.5961],
    [0.6706, 0.8667, 0.6431],
    [0.4000, 0.7608, 0.6471],
    [0.1961, 0.5333, 0.7412],
    [0.3686, 0.3098, 0.6353],
], dtype=np.float32)                                                            # My favorite colormap for visualization

class ImageFolderDataset(Dataset):
    """Simple image folder dataset used to avoid extra runtime dependencies."""

    def __init__(self, samples: Sequence[Tuple[Path, int]], image_size: int):
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, target = self.samples[index]

        # Load the image and convert it into three color channels
        image = Image.open(image_path).convert("RGB")

        # Resize images to fit the network input requirement without augmentation
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        # Transform image data into PyTorch channel first tensor format
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()

        return image_tensor, int(target)

class PConvLayer(nn.Module):
    """Partial Convolution layer for RHFNet."""

    def __init__(self, dim: int, num_div: int = 4):
        super().__init__()
        if dim % num_div != 0:
            raise ValueError("dim must be divisible by num_div for stable channel splitting")

        self.DimConv = dim // num_div                                      # Number of channels to apply convolution
        self.DimUntouched = dim - self.DimConv                             # Number of untouched channels

        self.conv = nn.Conv2d(
            in_channels=self.DimConv,
            out_channels=self.DimConv,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Initialize weights using He initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward input data through the layer at training and prediction time

        # Split the input along the channel dimension
        x1 = x[:, :self.DimConv, :, :]
        x2 = x[:, self.DimConv:, :, :]

        # Apply 3x3 spatial convolution to the split part x1
        x1_conv = self.conv(x1)

        # Concatenate the convolved channels with the untouched channels
        return torch.cat((x1_conv, x2), dim=1)

class RHFNetBlock(nn.Module):
    """Residual block used by each RHFNet stage."""

    def __init__(self, dim: int, num_div: int = 4):
        super().__init__()

        self.pconv = PConvLayer(dim, num_div)
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(dim * 2)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward input through the residual branch
        shortcut = x
        x = self.pconv(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.conv2(x)

        # Add the residual branch back to the shortcut branch
        return x + shortcut

class RHFNet(nn.Module):
    """RHFNet model based on FasterNet-T0 style embedding and stage depth."""

    def __init__(self, num_classes: int = 12):
        super().__init__()

        embed_dims = [40, 80, 160, 320]
        depths = [1, 2, 8, 2]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dims[0], kernel_size=4, stride=4, padding=0, bias=True),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
        )

        # Stages
        stages = []
        for stage_index, dim in enumerate(embed_dims):
            if stage_index > 0:
                stages.append(nn.Sequential(
                    nn.Conv2d(embed_dims[stage_index - 1], dim, kernel_size=2, stride=2, padding=0, bias=True),
                    nn.BatchNorm2d(dim),
                    nn.GELU(),
                ))

            for _ in range(depths[stage_index]):
                stages.append(RHFNetBlock(dim, num_div=4))

        self.stages = nn.Sequential(*stages)

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_conv = nn.Conv2d(embed_dims[-1], 1280, kernel_size=1, stride=1, padding=0, bias=True)
        self.head_gelu = nn.GELU()
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x)
        x = self.head_conv(x)
        x = self.head_gelu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class AverageMeter:
    """Running average meter for training and validation statistics."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, number: int) -> None:
        self.total += float(value) * int(number)
        self.count += int(number)

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

def set_random_seed(seed: int) -> None:
    # Fix random sources for reproducible splitting and training
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def find_image_samples(dataset_path: str) -> Tuple[List[Tuple[Path, int]], List[str]]:
    # Load images from the specified folder
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path was not found: {root}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    class_dirs = sorted([item for item in root.iterdir() if item.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders were found under: {root}")

    class_names = [item.name for item in class_dirs]
    class_to_index: Dict[str, int] = {class_name: index for index, class_name in enumerate(class_names)}

    samples: List[Tuple[Path, int]] = []
    for class_dir in class_dirs:
        class_index = class_to_index[class_dir.name]
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in extensions:
                samples.append((image_path, class_index))

    if not samples:
        raise RuntimeError(f"No image files were found under: {root}")

    return samples, class_names

def split_each_label(
    samples: Sequence[Tuple[Path, int]],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    # 8:2 split for Train and Validation sets
    rng = np.random.default_rng(seed)
    samples_by_label: Dict[int, List[Tuple[Path, int]]] = {}

    for image_path, target in samples:
        samples_by_label.setdefault(target, []).append((image_path, target))

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []

    for target in sorted(samples_by_label.keys()):
        label_samples = samples_by_label[target]
        indices = np.arange(len(label_samples))
        rng.shuffle(indices)

        train_count = int(np.floor(len(indices) * train_ratio))
        if len(indices) > 1:
            train_count = min(max(train_count, 1), len(indices) - 1)

        train_indices = indices[:train_count]
        val_indices = indices[train_count:]

        train_samples.extend([label_samples[int(index)] for index in train_indices])
        val_samples.extend([label_samples[int(index)] for index in val_indices])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples

def build_dataloaders(config: RHFNetConfig) -> Tuple[DataLoader, DataLoader, List[str]]:
    print("Start loading dataset and spliting...")

    samples, class_names = find_image_samples(config.Dataset_Path)
    if len(class_names) != config.Num_Classes:
        raise RuntimeError(
            f"The dataset contains {len(class_names)} classes, but Num_Classes is set to {config.Num_Classes}"
        )

    train_samples, val_samples = split_each_label(samples, config.Train_Ratio, config.Random_Seed)

    train_dataset = ImageFolderDataset(train_samples, config.Image_Size)
    val_dataset = ImageFolderDataset(val_samples, config.Image_Size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.Batch_Size,
        shuffle=True,
        num_workers=config.Num_Workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.Batch_Size,
        shuffle=False,
        num_workers=config.Num_Workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    return train_loader, val_loader, class_names

def build_optimizer(model: nn.Module, config: RHFNetConfig) -> torch.optim.Optimizer:
    # Construct the optimizer used for network training
    if config.Optimizer_Name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.Initial_LR, weight_decay=config.Weight_Decay)
    if config.Optimizer_Name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.Initial_LR, weight_decay=config.Weight_Decay)
    raise ValueError("Optimizer_Name must be adam or adamw")

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # Compute classification accuracy from raw network logits
    predictions = torch.argmax(logits, dim=1)
    correct = torch.sum(predictions == targets).item()
    return correct / targets.numel() * 100.0

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    # Train the model for one epoch
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        batch_acc = accuracy_from_logits(logits.detach(), targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(batch_acc, batch_size)

        progress.set_postfix(loss=f"{loss_meter.average:.4f}", acc=f"{acc_meter.average:.2f}%")

    return loss_meter.average, acc_meter.average

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    # Evaluate the model on the validation set
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    progress = tqdm(loader, desc=f"Epoch {epoch} Validation", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        predictions = torch.argmax(logits, dim=1)

        batch_size = images.size(0)
        batch_acc = accuracy_from_logits(logits, targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(batch_acc, batch_size)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        progress.set_postfix(loss=f"{loss_meter.average:.4f}", acc=f"{acc_meter.average:.2f}%")

    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)

    return loss_meter.average, acc_meter.average, y_true, y_pred

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: RHFNetConfig,
    class_names: Sequence[str],
    epoch: int,
    val_loss: float,
    val_acc: float,
    output_dir: Path,
) -> None:
    # Save the best validation network
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": list(class_names),
        "config": config.__dict__,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    torch.save(checkpoint, output_dir / "RHFNet_best.pth")

def plot_training_curves(history: Dict[str, List[float]], config: RHFNetConfig, output_dir: Path) -> None:
    # Plotting Accuracy and Loss Curves
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig_loss = plt.figure(num="Training & Validation Loss", figsize=(7, 5))
    ax_loss = fig_loss.add_subplot(111)
    ax_loss.grid(True)
    ax_loss.plot(epochs, history["train_loss"], color=JoeyBG_Colormap[-2], linewidth=2, label="Training Loss")
    ax_loss.plot(epochs, history["val_loss"], color=JoeyBG_Colormap[1], linewidth=2, label="Validation Loss")
    ax_loss.set_xlabel("Training epochs", fontname=config.Font_Name, fontsize=config.Font_Size_Axis, fontweight=config.Font_Weight_Axis)
    ax_loss.set_ylabel("Cross Entropy Loss", fontname=config.Font_Name, fontsize=config.Font_Size_Axis, fontweight=config.Font_Weight_Axis)
    ax_loss.tick_params(labelsize=config.Font_Size_Basis)
    ax_loss.legend(loc="upper right")
    fig_loss.tight_layout()
    fig_loss.savefig(output_dir / "RHFNet_Loss_Curve.png", dpi=300)

    fig_acc = plt.figure(num="Training & Validation Accuracy", figsize=(7, 5))
    ax_acc = fig_acc.add_subplot(111)
    ax_acc.grid(True)
    ax_acc.plot(epochs, history["train_acc"], color=JoeyBG_Colormap[-2], linewidth=2, label="Training Accuracy")
    ax_acc.plot(epochs, history["val_acc"], color=JoeyBG_Colormap[1], linewidth=2, label="Validation Accuracy")
    ax_acc.set_xlabel("Training epochs", fontname=config.Font_Name, fontsize=config.Font_Size_Axis, fontweight=config.Font_Weight_Axis)
    ax_acc.set_ylabel("Accuracy (%)", fontname=config.Font_Name, fontsize=config.Font_Size_Axis, fontweight=config.Font_Weight_Axis)
    ax_acc.set_ylim([0, 100])
    ax_acc.tick_params(labelsize=config.Font_Size_Basis)
    ax_acc.legend(loc="lower right")
    fig_acc.tight_layout()
    fig_acc.savefig(output_dir / "RHFNet_Accuracy_Curve.png", dpi=300)

    plt.close(fig_loss)
    plt.close(fig_acc)

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    config: RHFNetConfig,
    output_dir: Path,
) -> None:
    # Plotting Validation Confusion Matrix
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels).T

    class_labels = [f"S{index + 1}" for index in range(len(class_names))]

    fig_conf = plt.figure(num="Confusion Matrix Validation", figsize=(8, 8))
    ax_conf = fig_conf.add_subplot(111)
    image = ax_conf.imshow(matrix, cmap="Blues")
    fig_conf.colorbar(image, ax=ax_conf, fraction=0.046, pad=0.04)

    ax_conf.set_xticks(labels)
    ax_conf.set_yticks(labels)
    ax_conf.set_xticklabels(class_labels, fontname=config.Font_Name, fontsize=config.Font_Size_Basis, fontweight=config.Font_Weight_Basis)
    ax_conf.set_yticklabels(class_labels, fontname=config.Font_Name, fontsize=config.Font_Size_Basis, fontweight=config.Font_Weight_Basis)
    ax_conf.set_xlabel("Target Class", fontname=config.Font_Name, fontsize=config.Font_Size_Axis, fontweight=config.Font_Weight_Axis)
    ax_conf.set_ylabel("Output Class", fontname=config.Font_Name, fontsize=config.Font_Size_Axis, fontweight=config.Font_Weight_Axis)

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            text_color = "white" if matrix[row, col] > threshold else "black"
            ax_conf.text(col, row, str(matrix[row, col]), ha="center", va="center", color=text_color, fontsize=12)

    fig_conf.tight_layout()
    fig_conf.savefig(output_dir / "RHFNet_Confusion_Matrix.png", dpi=300)
    plt.close(fig_conf)

def count_trainable_parameters(model: nn.Module) -> int:
    # Count learnable parameters in the constructed network model
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

def main() -> None:
    config = RHFNetConfig()
    set_random_seed(config.Random_Seed)
    if config.Torch_Num_Threads > 0:
        torch.set_num_threads(config.Torch_Num_Threads)

    output_dir = Path(config.Output_Dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(config)

    print("Constructing RHFNet architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RHFNet(num_classes=config.Num_Classes).to(device)
    print(f"Execution device: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_metric = float("inf") if config.Save_Best_By == "val_loss" else -float("inf")
    best_y_true = np.array([], dtype=np.int64)
    best_y_pred = np.array([], dtype=np.int64)

    print("Start Training Process...")
    for epoch in range(1, config.Max_Epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, y_true, y_pred = validate_one_epoch(model, val_loader, criterion, device, epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:03d}/{config.Max_Epochs:03d}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Accuracy: {train_acc:.2f}% "
            f"Validation Loss: {val_loss:.4f} "
            f"Validation Accuracy: {val_acc:.2f}%"
        )

        current_metric = val_loss if config.Save_Best_By == "val_loss" else val_acc
        improved = current_metric < best_metric if config.Save_Best_By == "val_loss" else current_metric > best_metric

        if improved:
            best_metric = current_metric
            best_y_true = y_true
            best_y_pred = y_pred
            save_checkpoint(model, optimizer, config, class_names, epoch, val_loss, val_acc, output_dir)
            print("Best validation network has been updated.")

    print("Training Completed.")

    print("Plotting Accuracy, Loss Curves and Confusion Matrix...")
    plot_training_curves(history, config, output_dir)
    plot_confusion_matrix(best_y_true, best_y_pred, class_names, config, output_dir)

    np.savez(
        output_dir / "RHFNet_training_history.npz",
        train_loss=np.array(history["train_loss"], dtype=np.float32),
        train_acc=np.array(history["train_acc"], dtype=np.float32),
        val_loss=np.array(history["val_loss"], dtype=np.float32),
        val_acc=np.array(history["val_acc"], dtype=np.float32),
    )

    print(f"Results saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
