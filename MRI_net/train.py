import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

from . import config


def evaluate(model, loader, cls_criterion, recon_criterion, device: torch.device = config.DEVICE) -> dict:
    model.eval()
    correct, total = 0, 0
    total_cls_loss, total_recon_loss = 0.0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, recon = model(images)
            cls_loss = cls_criterion(logits, labels)
            recon_loss = recon_criterion(recon, images)
            total_cls_loss += cls_loss.item() * images.size(0)
            total_recon_loss += recon_loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return {
        "accuracy": correct / total,
        "cls_loss": total_cls_loss / total,
        "recon_loss": total_recon_loss / total,
    }


def train_model(
    model,
    train_loader,
    test_loader,
    class_names: list,
    device: torch.device = config.DEVICE,
    epochs: int = config.EPOCHS,
    checkpoint_path: str = config.CHECKPOINT_PATH,
) -> dict:
    cls_criterion = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_acc": [], "val_acc": [], "train_cls_loss": [], "train_recon_loss": []}
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_cls_loss, running_recon_loss, running_correct, running_total = 0.0, 0.0, 0, 0
        start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, recon = model(images)
            cls_loss = cls_criterion(logits, labels)
            recon_loss = recon_criterion(recon, images)  #removed .detach() from here
            loss = config.CLS_LOSS_WEIGHT * cls_loss + config.RECON_LOSS_WEIGHT * recon_loss
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            running_cls_loss += cls_loss.item() * images.size(0)
            running_recon_loss += recon_loss.item() * images.size(0)

        train_acc = running_correct / running_total
        train_cls_loss = running_cls_loss / running_total
        train_recon_loss = running_recon_loss / running_total
        val_metrics = evaluate(model, test_loader, cls_criterion, recon_criterion, device=device)
        scheduler.step(val_metrics["accuracy"])

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_cls_loss"].append(train_cls_loss)
        history["train_recon_loss"].append(train_recon_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | train_acc: {train_acc:.4f} cls_loss: {train_cls_loss:.4f} "
            f"recon_loss: {train_recon_loss:.4f} | val_acc: {val_metrics['accuracy']:.4f} "
            f"| time: {time.time() - start:.1f}s"
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save({"model_state": best_state, "class_names": class_names, "val_acc": best_acc}, checkpoint_path)
            print(f"  -> new best val_acc {best_acc:.4f}, checkpoint saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                print(f"No improvement for {config.EARLY_STOP_PATIENCE} epochs, stopping early.")
                break

    model.load_state_dict(best_state)
    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
    return history