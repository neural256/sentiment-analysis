import json
import torch

from src.config import Config
from src.model import TransformerSentimentClassifier
from src.preprocess import clean_text, tokenize


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def encode_new_tweet(text, token_to_idx, max_sequence_length):
    """
    Convert a new raw tweet into the same numerical format used during training.

    Story:
    raw tweet
    → clean text
    → tokenize
    → add <cls>
    → convert tokens to IDs
    → pad/truncate
    → tensor
    """

    pad_idx = token_to_idx["<pad>"]
    unk_idx = token_to_idx["<unk>"]
    cls_idx = token_to_idx["<cls>"]

    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)

    token_ids = [cls_idx]

    for token in tokens:
        token_id = token_to_idx.get(token, unk_idx)
        token_ids.append(token_id)

    if len(token_ids) > max_sequence_length:
        token_ids = token_ids[:max_sequence_length]

    if len(token_ids) < max_sequence_length:
        token_ids = token_ids + [pad_idx] * (max_sequence_length - len(token_ids))

    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    return input_tensor, cleaned_text, tokens


def predict_sentiment(text):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_to_idx = load_json(config.vocab_path)
    class_mapping = load_json(config.class_mapping_path)

    idx_to_class = {
        int(class_id): class_name
        for class_name, class_id in class_mapping.items()
    }

    num_classes = len(class_mapping)

    model = TransformerSentimentClassifier(
        vocab_size=len(token_to_idx),
        embedding_dim=config.embedding_dim,
        num_heads=config.num_attention_heads,
        num_encoder_layers=config.num_encoder_layers,
        feedforward_dim=config.feedforward_dim,
        num_classes=num_classes,
        max_sequence_length=config.max_sequence_length,
        dropout=config.dropout,
        pad_idx=token_to_idx["<pad>"],
    )

    model.load_state_dict(
        torch.load(config.best_model_path, map_location=device)
    )

    model = model.to(device)
    model.eval()

    input_tensor, cleaned_text, tokens = encode_new_tweet(
        text=text,
        token_to_idx=token_to_idx,
        max_sequence_length=config.max_sequence_length,
    )

    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class_id = torch.argmax(probabilities, dim=1).item()

    predicted_label = idx_to_class[predicted_class_id]
    confidence = probabilities[0, predicted_class_id].item()

    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "all_probabilities": {
            idx_to_class[i]: probabilities[0, i].item()
            for i in range(num_classes)
        },
    }

if __name__ == "__main__":
    tweet = input("Enter an airline tweet: ")

    result = predict_sentiment(tweet)

    print("\nPrediction result")
    print("-" * 50)
    print("Original text:", result["original_text"])
    print("Cleaned text:", result["cleaned_text"])
    print("Tokens:", result["tokens"])
    print("Predicted sentiment:", result["predicted_label"])
    print("Confidence:", round(result["confidence"], 4))

    print("\nClass probabilities:")
    for label, probability in result["all_probabilities"].items():
        print(f"{label}: {probability:.4f}")