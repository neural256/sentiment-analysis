import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import TweetSentimentDataset
from src.evaluate import evaluate_on_test_set, save_test_report
from src.load_data import load_airline_tweets_from_kaggle, load_airline_tweets_from_openml
from src.model import TransformerSentimentClassifier
from src.plots import plot_confusion_matrix, plot_training_history
from src.preprocess import Vocabulary, clean_text, texts_to_padded_ids, tokenize
from src.train import fit_model
from src.utils import compute_class_weights, create_directories, save_json, set_seed


def main():
    config = Config()
    create_directories(config)
    set_seed(config.random_seed)
    print("=" * 80)
    print("AIRLINE TWEETS SENTIMENT ANALYSIS USING TRANSFORMER ENCODER")
    print("=" * 80)

    df = load_airline_tweets_from_openml(config.openml_data_path)

    print("\nDataset preview:")
    print(df.head())

    print("\nClass distribution:")
    print(df["label"].value_counts())

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["label"].astype(str))
    class_names = list(label_encoder.classes_)

    class_mapping = {class_name: int(i) for i, class_name in enumerate(class_names)}
    print("\nClass mapping:")
    print(class_mapping)
    save_json(class_mapping, config.class_mapping_path)

    texts = df["text"].astype(str).values

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        texts,
        labels,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=labels,
    )

    val_fraction = config.val_size / (1.0 - config.test_size)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction,
        random_state=config.random_seed,
        stratify=y_train_val,
    )

    print("\nSplit sizes:")
    print(f"Training:   {len(x_train)}")
    print(f"Validation: {len(x_val)}")
    print(f"Test:       {len(x_test)}")

    train_tokens = [tokenize(clean_text(text)) for text in x_train]
    val_tokens = [tokenize(clean_text(text)) for text in x_val]
    test_tokens = [tokenize(clean_text(text)) for text in x_test]

    vocab = Vocabulary(
        max_size=config.max_vocab_size,
        min_freq=config.min_word_frequency,
    )
    vocab.build(train_tokens)

    print("\nVocabulary size:", len(vocab))
    save_json(vocab.token_to_idx, config.vocab_path)

    x_train_ids = texts_to_padded_ids(
        train_tokens,
        vocab=vocab,
        max_length=config.max_sequence_length,
        add_cls=True,
    )
    x_val_ids = texts_to_padded_ids(
        val_tokens,
        vocab=vocab,
        max_length=config.max_sequence_length,
        add_cls=True,
    )
    x_test_ids = texts_to_padded_ids(
        test_tokens,
        vocab=vocab,
        max_length=config.max_sequence_length,
        add_cls=True,
    )

    train_dataset = TweetSentimentDataset(x_train_ids, y_train)
    val_dataset = TweetSentimentDataset(x_val_ids, y_val)
    test_dataset = TweetSentimentDataset(x_test_ids, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    model = TransformerSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        num_heads=config.num_attention_heads,
        num_encoder_layers=config.num_encoder_layers,
        feedforward_dim=config.feedforward_dim,
        num_classes=len(class_names),
        max_sequence_length=config.max_sequence_length,
        dropout=config.dropout,
        pad_idx=vocab.pad_idx,
    ).to(device)

    print("\nModel architecture:")
    print(model)

    if config.use_weighted_loss:
        class_weights = compute_class_weights(y_train, num_classes=len(class_names))
        class_weights = class_weights.to(device)
        print("\nUsing weighted CrossEntropyLoss.")
        print("Class weights:", class_weights.detach().cpu().numpy())
    else:
        class_weights = None
        print("\nUsing standard CrossEntropyLoss.")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config.num_epochs,
        grad_clip=config.grad_clip,
        best_model_path=config.best_model_path,
        patience=config.patience,
    )

    model.load_state_dict(torch.load(config.best_model_path, map_location=device))

    test_metrics = evaluate_on_test_set(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
    )

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Test loss:        {test_metrics['loss']:.4f}")
    print(f"Accuracy:         {test_metrics['accuracy']:.4f}")
    print(f"Macro precision:  {test_metrics['macro_precision']:.4f}")
    print(f"Macro recall:     {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1-score:   {test_metrics['macro_f1']:.4f}")

    print("\nClassification report:")
    print(test_metrics["classification_report_text"])

    save_test_report(test_metrics, config.test_metrics_path)

    plot_training_history(
        history,
        loss_plot_path=config.loss_plot_path,
        accuracy_plot_path=config.accuracy_plot_path,
        f1_plot_path=config.f1_plot_path,
    )

    plot_confusion_matrix(
        confusion_matrix_array=np.array(test_metrics["confusion_matrix"]),
        class_names=class_names,
        output_path=config.confusion_matrix_plot_path,
    )

    print("\nSaved outputs:")
    print(f"Best model:       {config.best_model_path}")
    print(f"Loss plot:        {config.loss_plot_path}")
    print(f"Accuracy plot:    {config.accuracy_plot_path}")
    print(f"F1 plot:          {config.f1_plot_path}")
    print(f"Confusion matrix: {config.confusion_matrix_plot_path}")
    print(f"Test metrics:     {config.test_metrics_path}")


if __name__ == "__main__":
    main()
