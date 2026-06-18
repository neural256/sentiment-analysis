from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    kaggle_dataset_name: str = "crowdflower/twitter-airline-sentiment"
    random_seed: int = 42

    test_size: float = 0.15
    val_size: float = 0.15

    max_vocab_size: int = 10000
    min_word_frequency: int = 1
    max_sequence_length: int = 50

    embedding_dim: int = 128
    num_attention_heads: int = 4
    num_encoder_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.3

    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 25
    grad_clip: float = 1.0
    use_weighted_loss: bool = True

    project_root: Path = Path(__file__).resolve().parents[1]

    @property
    def raw_data_path(self) -> Path:
        return self.project_root / "data" / "raw" / "kaggle_airline_tweets.csv"

    @property
    def best_model_path(self) -> Path:
        return self.project_root / "outputs" / "models" / "best_model.pt"

    @property
    def loss_plot_path(self) -> Path:
        return self.project_root / "outputs" / "plots" / "loss_curve.png"

    @property
    def accuracy_plot_path(self) -> Path:
        return self.project_root / "outputs" / "plots" / "accuracy_curve.png"

    @property
    def confusion_matrix_plot_path(self) -> Path:
        return self.project_root / "outputs" / "plots" / "confusion_matrix.png"

    @property
    def test_metrics_path(self) -> Path:
        return self.project_root / "outputs" / "reports" / "test_metrics.json"

    @property
    def class_mapping_path(self) -> Path:
        return self.project_root / "outputs" / "reports" / "class_mapping.json"

    @property
    def vocab_path(self) -> Path:
        return self.project_root / "outputs" / "reports" / "vocab.json"
