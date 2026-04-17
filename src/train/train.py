from src.utils import logger
from src.loss import get_loss

import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm


def train(config, model, train_loader, val_loader):

    #init logger
    log = logger(log_dir=config.log_path, 
                 log_filename=f"log_{config.experiment_name}.log")
    
    log.info(f"Experiment Name: {config.experiment_name}")
    log.info("Starting training pipeline...")

    #device setup (GPU if available)
    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")
    log.info(f"Using device: {device}")

    #seed
    if config.seed_flag:
        seed = 42
        torch.manual_seed(seed)
        if cuda_flag:
            torch.cuda.manual_seed_all(seed)
        log.info(f"Seed is set to {seed}")

    #init model
    model.to(device)

    #init loss and optimizer
    criterion = get_loss(config.training['loss'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training['lr'])

    log.info(f"Loss: {criterion}")
    log.info(f"LR: {config.training['lr']}")

    #init tensorBoard
    tensorboard_log_path = os.path.join(config.log_path, "tensorboard")
    os.makedirs(tensorboard_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_path)

    #training/Validation Loop
    for epoch in range(config.training["epochs"]):
        model.train()
        train_loss = 0.0

        # ----- Train loop -----
        for images, targets in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", ncols=100):
            images = images.to(device)
            targets = targets.to(device)  # shape [B,1]

            logits = model(images) #model output shape [B,1]

            loss = criterion(logits, targets) #compute loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)

        # ----- Test loop -----
        model.eval()
        val_loss = 0.0
        TP, FP, FN = 0, 0, 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device) #shape [B,1]

                logits = model(images) #model output shape [B,1]

                loss = criterion(logits, targets)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                TP += ((preds == 1) & (targets == 1)).sum().item()
                FP += ((preds == 1) & (targets == 0)).sum().item()
                FN += ((preds == 0) & (targets == 1)).sum().item()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        avg_val_loss = val_loss / len(val_loader)

        # ---- TensorBoard ----
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
        writer.add_scalar('Metrics/Precision', precision, epoch + 1)
        writer.add_scalar('Metrics/Recall', recall, epoch + 1)

        # ---- Logging ----
        log.info(
            f"Epoch {epoch+1}/{config.training['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f}"
        )

        # ----- Save -----
        if (epoch + 1) % config.training['save_every'] == 0:
            model_path = os.path.join(config.log_path, "models")
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, f"model_epoch_{epoch+1}.pth"))

    #final save
    model_path = os.path.join(config.log_path, "models")
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, "final_model.pth"))

    writer.close()
    log.info("Training completed successfully.")