import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def predict(model, data_loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    return all_labels, all_predictions


def compute_average_loss(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(data_loader.dataset)


def evaluate_on_test_set(
    model,
    data_loader,
    criterion,
    device,
    class_names,
):
    loss = compute_average_loss(model, data_loader, criterion, device)
    y_true, y_pred = predict(model, data_loader, device)

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report_text": report_text,
    }


def save_test_report(metrics, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
