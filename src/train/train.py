from src.utils import logger
from src.loss import get_loss

import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm


def compute_segmentation_metrics(logits, targets, threshold=0.5, eps=1e-8):
    """
    logits:  [B, 1, H, W]
    targets: [B, 1, H, W]
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # Flatten per batch
    preds = preds.reshape(preds.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum - intersection

    dice = ((2 * intersection + eps) / (pred_sum + target_sum + eps)).mean().item() # average the metric across all images in the batch
    iou = ((intersection + eps) / (union + eps)).mean().item() # average the metric across all images in the batch

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp + eps) # pixel precision
    recall = tp / (tp + fn + eps) # pixel recall

    return dice, iou, precision, recall


def train(config, model, train_loader, val_loader):
    # init logger
    log = logger(
        log_dir=config.log_path,
        log_filename=f"log_{config.experiment_name}.log")

    log.info(f"Experiment Name: {config.experiment_name}")
    log.info("Starting training pipeline...")

    # device setup
    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")
    log.info(f"Using device: {device}")

    log.info(f"Train dataset size: {len(train_loader.dataset)}")
    log.info(f"Validation dataset size: {len(val_loader.dataset)}")
    log.info(f"Train batches per epoch: {len(train_loader)}")
    log.info(f"Validation batches per epoch: {len(val_loader)}")

    # seed
    if config.seed_flag:
        seed = 42
        torch.manual_seed(seed)
        if cuda_flag:
            torch.cuda.manual_seed_all(seed)
        log.info(f"Seed is set to {seed}")

    # init model
    model.to(device)

    # init loss and optimizer
    criterion = get_loss(config.training['loss'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training['lr'])

    log.info(f"Loss: {criterion}")
    log.info(f"LR: {config.training['lr']}")

    # init TensorBoard
    tensorboard_log_path = os.path.join(config.log_path, "tensorboard")
    os.makedirs(tensorboard_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_path)

    best_val_dice = -1.0 #initialize

    # training / validation loop
    for epoch in range(config.training["epochs"]):
        # ----- Train loop -----
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        train_precision = 0.0
        train_recall = 0.0

        for images, masks in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", ncols=100):
            images = images.to(device)   # [B, 3, H, W]
            masks = masks.to(device)     # [B, 1, H, W]

            logits = model(images)       # expected: [B, 1, H, W]
            loss = criterion(logits, masks) # compute loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            dice, iou, precision, recall = compute_segmentation_metrics(logits, masks)
            train_dice += dice
            train_iou += iou
            train_precision += precision
            train_recall += recall

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_train_precision = train_precision / len(train_loader)
        avg_train_recall = train_recall / len(train_loader)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Metrics_Train/Dice", avg_train_dice, epoch + 1)
        writer.add_scalar("Metrics_Train/IoU", avg_train_iou, epoch + 1)
        writer.add_scalar("Metrics_Train/Precision", avg_train_precision, epoch + 1)
        writer.add_scalar("Metrics_Train/Recall", avg_train_recall, epoch + 1)

        # ----- Validation loop -----
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_precision = 0.0
        val_recall = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()

                dice, iou, precision, recall = compute_segmentation_metrics(logits, masks)
                val_dice += dice
                val_iou += iou
                val_precision += precision
                val_recall += recall

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_precision = val_precision / len(val_loader)
        avg_val_recall = val_recall / len(val_loader)

        # ----- TensorBoard -----
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
        writer.add_scalar("Metrics_Validation/Dice", avg_val_dice, epoch + 1)
        writer.add_scalar("Metrics_Validation/IoU", avg_val_iou, epoch + 1)
        writer.add_scalar("Metrics_Validation/Precision", avg_val_precision, epoch + 1)
        writer.add_scalar("Metrics_Validation/Recall", avg_val_recall, epoch + 1)

        # ----- Logging -----
        log.info(
            f"Epoch {epoch+1}/{config.training['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Dice: {avg_train_dice:.4f} | "
            f"Train Precision: {avg_train_precision:.4f} | "
            f"Train Recall: {avg_train_recall:.4f} | " 
            f"Train IoU: {avg_train_iou:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f} | "
            f"Val IoU: {avg_val_iou:.4f} | "
            f"Val Precision: {avg_val_precision:.4f} | "
            f"Val Recall: {avg_val_recall:.4f}")

        # ----- Periodic save -----
        if (epoch + 1) % config.training['save_every'] == 0:
            model_path = os.path.join(config.log_path, "models")
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, f"model_epoch_{epoch+1}.pth"))

        # ----- Best model save based on validation Dice -----
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            model_path = os.path.join(config.log_path, "models")
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, "best_model.pth"))
            log.info(f"Best model updated at epoch {epoch+1} with Val Dice: {avg_val_dice:.4f}")

    # final save
    model_path = os.path.join(config.log_path, "models")
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, "final_model.pth"))

    writer.close()
    log.info("Training completed successfully.")