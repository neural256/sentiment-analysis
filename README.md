# Airline Tweets Sentiment Analysis Using a Transformer Encoder

This project solves Task 1 for a Deep Neural Networks course project.

Dataset:
- OpenML dataset ID: 43397
- Dataset name: Airlines-Tweets-Sentiments
- Task: multiclass sentiment classification
- Input: airline-related tweet text
- Output: sentiment class

Architecture:

Raw tweet
→ cleaning
→ tokenization
→ vocabulary
→ token IDs
→ padding/truncation
→ embedding
→ positional encoding
→ Transformer Encoder
→ CLS representation
→ fully connected classifier
→ sentiment prediction

This project uses a Transformer Encoder, not a full encoder-decoder Transformer. A decoder is not needed because sentiment analysis is classification, not text generation.

## Run

```bash
pip install -r requirements.txt
python main.py
```

The first run needs internet access because the dataset is downloaded from OpenML.

## Outputs

After training, the project saves:

- outputs/models/best_model.pt
- outputs/plots/loss_curve.png
- outputs/plots/accuracy_curve.png
- outputs/plots/confusion_matrix.png
- outputs/reports/test_metrics.json
- outputs/reports/class_mapping.json
- outputs/reports/vocab.json
