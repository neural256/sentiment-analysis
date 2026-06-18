from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    grad_clip: float = 1.0,
):
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for input_ids, labels in tqdm(data_loader, desc="Training", leave=False):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


def evaluate_one_epoch(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in tqdm(data_loader, desc="Validation", leave=False):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


def fit_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs: int,
    grad_clip: float,
    best_model_path,
) -> Dict[str, List[float]]:
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
    }

    best_val_macro_f1 = -1.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
        )

        val_loss, val_acc, val_f1 = evaluate_one_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["train_macro_f1"].append(train_f1)
        history["val_macro_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"val acc: {val_acc:.4f} | "
            f"train F1: {train_f1:.4f} | "
            f"val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved new best model with validation macro-F1 = {val_f1:.4f}")

    return history
