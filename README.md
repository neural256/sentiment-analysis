# Airline Tweets Sentiment Analysis Using a Transformer Encoder

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

## Main idea

This project performs supervised sentiment classification on airline tweets. The input is a tweet, and the output is a sentiment class.

## Why Transformer Encoder?

A full Transformer has an encoder and a decoder. The decoder is useful when the model must generate a sequence, such as in translation or text generation. In this project, the model only needs to read a tweet and classify it into one sentiment class. Therefore, a Transformer Encoder is sufficient.

## Why preprocessing?

Raw text cannot be processed directly by a neural network. Therefore, the text is cleaned, tokenized, converted into integer IDs, and padded or truncated.

## Why embeddings?

Token IDs are just integer labels. The embedding layer converts each token ID into a dense vector that the model can learn from.

## Why positional encoding?

Self-attention does not naturally know word order. Positional encoding is added so that the model knows where each token appears in the tweet.

## Why CLS token?

The Transformer Encoder gives an output vector for every token. Sentiment classification needs one vector for the whole tweet. Therefore, a special <cls> token is placed at the beginning of every tweet, and its final output representation is used as the tweet representation.

## Why self-attention?

Self-attention allows words to look at other words in the same tweet. This is important because words such as "not" can change the meaning of words such as "good".

## Why CrossEntropyLoss?

This is a multiclass classification problem. The model outputs logits for each class, and CrossEntropyLoss compares these logits with the correct class index.

## Why weighted loss?

Sentiment datasets can be imbalanced. Weighted loss gives more importance to minority classes so the model does not only favor the majority class.

## Why macro-F1?

Accuracy can be misleading when classes are imbalanced. Macro-F1 gives equal importance to each class.

## Outputs

After training, the project saves:

- outputs/models/best_model.pt
- outputs/plots/loss_curve.png
- outputs/plots/accuracy_curve.png
- outputs/plots/confusion_matrix.png
- outputs/reports/test_metrics.json
- outputs/reports/class_mapping.json
- outputs/reports/vocab.json
