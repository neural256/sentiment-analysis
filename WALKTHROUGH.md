# Your Project, Explained Like You Built It

A complete, ground-up walkthrough of the **Airline Tweets Sentiment Analysis** project.
No Python or ML background assumed. Read it top to bottom once, then keep it as a reference.

> This file is **notes only**. None of your actual code was changed. Delete this file anytime.

---

## 0. The one-sentence summary

You built a program that reads an airline tweet (e.g. *"@united lost my bag again"*) and predicts whether it is **negative, neutral, or positive** — using a small **Transformer**, the same family of model that powers ChatGPT, but tiny and trained by you.

---

## 1. The big picture (read this first)

Everything in the project is one **assembly line**. Text goes in one end, a prediction comes out the other. Each file is one station on the line:

```
A raw tweet (text)
   │
   ├─ load_data.py      get the tweets from Kaggle into a clean table
   ├─ preprocess.py     clean text → split into words → turn words into numbers → pad to equal length
   ├─ dataset.py        wrap those numbers so PyTorch can feed them in batches
   ├─ model.py          THE BRAIN: numbers → embeddings → Transformer → a guess
   ├─ train.py          show the brain thousands of examples so it learns
   ├─ evaluate.py       grade the trained brain on tweets it never saw
   ├─ plots.py          draw the learning curves and a confusion matrix
   ├─ utils.py          small shared helpers (randomness, folders, saving files)
   ├─ config.py         the control panel: every setting in one place
   │
   ├─ main.py           the conductor: runs all of the above in order (TRAINING)
   └─ predict.py        uses the finished brain on ONE new tweet (USING IT)
```

**Two ways to run the project:**
- `python main.py` → trains a brand-new model from scratch and saves it.
- `python predict.py` → loads the already-trained model and classifies a tweet you type in.

Hold this picture in your head. Every detail below hangs off it.

---

## 2. Concepts you need first (plain English)

You don't need to memorize these — just get the gist. We'll meet each one again in context.

### Machine-learning words

- **Model / network** — a big box of numbers (called **weights**) plus a recipe for combining them. At first the numbers are random, so it guesses garbage.
- **Training** — showing the model many examples and **slightly nudging** its numbers each time so its guesses get less wrong. That's all "learning" is: nudging numbers.
- **Loss** — a single number that measures *how wrong* a guess was. Big loss = very wrong. Training = "make the loss small."
- **Gradient / backpropagation** — the math that figures out *which direction* to nudge each weight to reduce the loss. PyTorch does this for you automatically.
- **Optimizer** — the rule that actually does the nudging (here: **Adam**). The **learning rate** is how big each nudge is.
- **Epoch** — one full pass through all the training tweets. You train for several epochs.
- **Batch** — you don't feed tweets one at a time; you feed them in small groups (here **32**) for speed and stability.
- **Embedding** — turning a word into a list of numbers (a **vector**) that captures its "meaning." Words used similarly end up with similar vectors. These vectors are *learned* during training.
- **Transformer / attention** — a design where, to understand a word, the model **looks at every other word** in the sentence and decides which ones matter. This is how it learns that *"not good"* is negative even though *"good"* appears.
- **Overfitting** — when the model memorizes the training tweets instead of learning general patterns, so it aces training but fails on new tweets. **Dropout** and **weight decay** fight this.
- **Train / validation / test split** — you split your data into three piles: **train** (to learn from), **validation** (to check progress and pick the best version), **test** (touched only once, for an honest final grade).

### Python words

- **Variable** — a name for a value: `x = 5`.
- **Function** — a reusable recipe: `def clean_text(text): ...`. You "call" it: `clean_text("hi")`.
- **Class** — a blueprint that bundles data + functions together. `Vocabulary` is a class; from it you make an **object** (an actual vocabulary).
- **`__init__`** — the "setup" function that runs when you create an object. It fills in the starting data.
- **`self`** — inside a class, the word for "this particular object." `self.max_size` = "this object's max_size."
- **Type hints** — the `: int`, `: str`, `-> Path` annotations. They're **documentation for humans**; Python doesn't enforce them. They just say "this is meant to be an integer," etc.
- **`@dataclass`, `@property`** — **decorators**, little tags above a function/class that add behavior. Explained where they appear (`config.py`).
- **List comprehension** — a compact one-line loop that builds a list: `[w.lower() for w in words]` means "lowercase every word."
- **`Path`** — an object that represents a file location, and handles Windows vs Mac/Linux slashes for you.

That's enough to read every file. Let's go station by station, in the order the data flows.

---

## 3. `config.py` — the control panel

**Its job:** hold *every* setting in one place so you never hunt through code to change a number.

```python
@dataclass
class Config:
    random_seed: int = 42
    embedding_dim: int = 128
    num_epochs: int = 25
    ...
```

**What `@dataclass` does:** Normally, to make a class that just stores settings, you'd write a lot of boilerplate. The `@dataclass` decorator writes that boilerplate **for you**. You just list the fields and their defaults. That's why this class has no visible `__init__` — the decorator generated it.

**The settings, grouped:**

| Group | Setting | Meaning |
|---|---|---|
| Data | `test_size=0.15`, `val_size=0.15` | 15% test, 15% validation, the rest (~70%) train |
| Vocabulary | `max_vocab_size=10000` | keep at most the 10,000 most common words |
| | `max_sequence_length=50` | every tweet becomes exactly 50 tokens long |
| Model | `embedding_dim=128` | each word becomes 128 numbers |
| | `num_attention_heads=4` | the Transformer "looks" 4 different ways at once |
| | `num_encoder_layers=2` | stack 2 Transformer layers |
| | `feedforward_dim=256` | size of the internal processing layer |
| | `dropout=0.3` | randomly silence 30% of signals while training (anti-overfit) |
| Training | `batch_size=32`, `learning_rate=1e-3` | 32 tweets per step; nudge size 0.001 |
| | `num_epochs=25` | go through the data 25 times |
| | `use_weighted_loss=True` | compensate for the data being mostly negative tweets |

**The `@property` methods at the bottom:**

```python
@property
def best_model_path(self) -> Path:
    return self.project_root / "outputs" / "models" / "best_model.pt"
```

A `@property` is a method you read **like a variable** — you write `config.best_model_path`, *not* `config.best_model_path()`. Why do it this way? So all output locations are **computed from one root folder**. `project_root` is found automatically with `Path(__file__).resolve().parents[1]` = "the folder two levels up from this file." This is why the project works no matter where on your computer it lives — nothing is hard-coded to `D:\pycharm\...`.

**Why this design:** One config object gets passed around everywhere. Change `num_epochs` here, and the whole project obeys.

**What else you could use:** `argparse` (settings from the command line, like `python main.py --epochs 50`), or a `YAML` file (settings in a separate text file). For a course project, a dataclass is the simplest clean choice — easy to defend.

---

## 4. `load_data.py` — getting the tweets

**Its job:** download the dataset from Kaggle and hand back a clean two-column table: `text` and `label`.

**The flow:**
1. `download_kaggle_dataset(...)` — uses the Kaggle API to download and unzip the dataset, but **first checks if it's already on disk** so you don't re-download every run. Smart.
2. It finds the `Tweets.csv` file inside the download.
3. `load_airline_tweets_from_kaggle(...)` reads that CSV with **pandas** (`pd.read_csv`). A pandas **DataFrame** is just a table, like a spreadsheet in code.
4. It keeps only two columns and **renames** them:
   - `text` (the tweet) stays `text`
   - `airline_sentiment` becomes `label`
5. It drops empty rows and saves a tidy copy.

**Why rename to `text` / `label`:** so the rest of the project never has to know the messy original column names. Clean boundary.

> **📝 Teaching note (something to improve later, not now):** there are **two almost identical functions**, `find_tweets_csv_if_exists` and `find_tweets_csv`. One returns `None` if the file is missing; the other raises an error. They could be merged into one. This is *duplicated code* — the kind of thing you'd tidy in a cleanup pass.

> **⚠️ Documentation bug:** your **README says the data comes from OpenML** (dataset 43397). But this code clearly downloads from **Kaggle** (`crowdflower/twitter-airline-sentiment`). The code is the truth; the README is stale. If anyone asks "where's your data from?", the right answer is **Kaggle**. Worth fixing the README.

**What else you could use:** `pandas.read_csv` straight from a URL, or the `datasets` library from Hugging Face. Kaggle is fine and standard.

---

## 5. `preprocess.py` — turning text into numbers

This is the heart of the "data" side. A neural network can't read letters — it only does math on numbers. This file is the **translator**.

### Step 1 — `clean_text`

```python
text = re.sub(r"http\S+|www\.\S+", f" {URL_TOKEN} ", text)   # links → <url>
text = re.sub(r"@\w+", f" {USER_TOKEN} ", text)              # @united → <user>
text = re.sub(r"#(\w+)", r"\1", text)                        # #fail → fail
```

`re` is Python's **regular expressions** library — pattern-matching for text. Here it:
- lowercases everything,
- replaces every web link with a single tag `<url>`,
- replaces every @mention with `<user>`,
- strips the `#` off hashtags (keeps the word).

**Why:** the *exact* URL or the *exact* username doesn't carry sentiment — but "there is a link here" or "this is aimed at someone" might. Collapsing them to one tag keeps the vocabulary small and meaningful.

> **📝 Note:** punctuation like `!` and `?` and emojis get dropped here. Those *do* carry sentiment ("great!!!" vs "great"). Not a bug — a simplicity trade-off. A good thing to mention as a known limitation.

### Step 2 — `tokenize`

Splits the cleaned string into a list of **tokens** (words/units): `"flight delayed"` → `["flight", "delayed"]`. The regex is ordered so it recognizes the special tags (`<url>`, `<user>`) as single tokens before splitting normal words.

### Step 3 — `Vocabulary` (a class)

A neural net needs each word as an **integer ID**. The vocabulary is the **dictionary** that maps `word → number` and back.

It starts with five **special tokens**, always at fixed IDs:

| ID | Token | Purpose |
|---|---|---|
| 0 | `<pad>` | filler to make all tweets the same length |
| 1 | `<unk>` | "unknown" — any word not in the vocabulary |
| 2 | `<url>` | stood in for links |
| 3 | `<user>` | stood in for @mentions |
| 4 | `<cls>` | a "summary slot" — **remember this one, it's important in the model** |

- `build(...)` counts every word in the **training** tweets and adds the most common ones, up to `max_vocab_size`. Rare words never get an ID — they'll become `<unk>`.
- `encode_tokens(...)` does the actual lookup: word → ID, using `<unk>` (ID 1) for anything unseen.

**Why a class and not loose functions?** Because the vocabulary is **state** (the word↔ID maps) plus **behavior** (build, encode) that belong together. That's exactly what classes are for.

### Step 4 — `texts_to_padded_ids` and `pad_or_truncate`

Models want every input the **same length**. So each tweet is forced to exactly `max_sequence_length` (50):
- too long → cut off the end (`truncate`),
- too short → fill with `<pad>` (ID 0) until it's 50.

And critically, with `add_cls=True`, a `<cls>` token (ID 4) is glued to the **front** of every tweet:

```python
ids = [vocab.cls_idx] + ids        # <cls> goes first
```

Hold that thought — the model reads exactly that front slot to make its decision.

**What else you could use:** real-world models use **subword tokenizers** (BPE/WordPiece) that split "unhappiness" into "un + happi + ness," so they never hit `<unk>`. Yours is simpler **word-level** tokenization — perfect for learning, and easy to explain.

---

## 6. `dataset.py` — handing data to PyTorch

**Its job:** wrap your lists of numbers in the format PyTorch's data loader expects.

```python
class TweetSentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
```

- A **tensor** is PyTorch's version of a list/grid of numbers — like a NumPy array, but it can run on a GPU and supports the gradient math. `dtype=torch.long` means "whole numbers" (correct for word IDs and class labels).
- PyTorch requires any dataset to answer two questions, so the class defines exactly two methods:
  - `__len__` → "how many examples are there?"
  - `__getitem__(idx)` → "give me example number `idx`" (returns one tweet + its label).

That's the entire contract. With those two methods, PyTorch's `DataLoader` can automatically shuffle and batch your data.

> **📝 Tiny note:** the length-mismatch check happens *after* building the tensors. Checking first would be marginally cleaner, but it's harmless.

---

## 7. `model.py` — the brain (the most important file)

This is where the actual intelligence lives. It has two classes.

### 7a. `PositionalEncoding`

**The problem it solves:** a Transformer looks at all words **simultaneously**, so by itself it has *no sense of word order*. "dog bites man" and "man bites dog" would look identical. That's unacceptable.

**The fix:** add a unique "position fingerprint" to each word's vector, built from **sine and cosine waves** of different frequencies:

```python
pe[:, 0::2] = torch.sin(position * div_term)   # even slots: sine
pe[:, 1::2] = torch.cos(position * div_term)   # odd slots: cosine
```

Now position 1 looks measurably different from position 7, so the model can tell where each word sits.

- **Why sine/cosine and not just 1, 2, 3, …?** The wave pattern keeps values in a tidy range and lets the model reason about **relative** distances. It's the exact scheme from the original "Attention Is All You Need" paper.
- **`register_buffer("pe", pe)`** — saves this table inside the model so it automatically moves to the GPU with the model, but is **not trained** (it's a fixed formula, not something to learn). That's the difference between a *buffer* and a *parameter*.

### 7b. `TransformerSentimentClassifier`

This is the full brain. Walk through `forward` — the method that runs when you call `model(input_ids)`:

```python
def forward(self, input_ids):
    padding_mask = input_ids.eq(self.pad_idx)          # 1) mark the filler
    x = self.embedding(input_ids)                      # 2) IDs → vectors
    x = x * math.sqrt(self.embedding_dim)              # 3) scale
    x = self.positional_encoding(x)                    # 4) add position info
    encoded = self.transformer_encoder(                # 5) the Transformer
        x, src_key_padding_mask=padding_mask)
    cls_representation = encoded[:, 0, :]              # 6) read the <cls> slot
    x = self.dropout(cls_representation)               # 7) classify
    x = self.fc1(x); x = self.relu(x); x = self.dropout(x)
    logits = self.output(x)                            # 8) one score per class
    return logits
```

Step by step:

1. **Padding mask** — finds which positions are `<pad>` filler, so the model can be told to **ignore** them. You don't want attention wasting effort on filler.
2. **Embedding** — `nn.Embedding` is a lookup table: each word ID → a learnable 128-number vector. `padding_idx=pad_idx` keeps `<pad>`'s vector at zero. **This table is one of the main things that gets trained.**
3. **Scaling by √128** — keeps the embedding numbers on a similar scale to the positional waves so neither drowns the other. Again, straight from the original paper.
4. **Positional encoding** — adds the position fingerprints from 7a.
5. **The Transformer encoder** — `nn.TransformerEncoder` with `num_encoder_layers=2`. This is **attention**: every token looks at every other token (except masked filler) and updates its own meaning based on context. After this, the token vectors are "context-aware." This is the part that learns "not good ≠ good."
   - `num_heads=4` → it does this looking **4 ways in parallel** ("attention heads"), each free to focus on a different kind of relationship.
   - The code checks `embedding_dim % num_heads == 0` in `__init__` and raises an error if not — because the 128 dims must split evenly across the 4 heads (32 each).
6. **Read the `<cls>` slot** — `encoded[:, 0, :]` grabs position 0. Remember from preprocessing: position 0 is always the `<cls>` token. After the encoder, that one slot has "absorbed" information from the whole tweet, so it serves as a **summary** of the entire tweet. This trick is borrowed from **BERT**.
7. **Classifier head** — a small standard neural net (`Linear → ReLU → Linear`) with **dropout** (randomly zeroing 30% of signals during training to prevent memorization). `ReLU` is just "keep positives, zero out negatives" — the standard nonlinearity.
8. **Logits** — the final output: three raw scores, one per class (negative/neutral/positive). The biggest score is the prediction. (They're turned into probabilities later, only when needed.)

**Why a Transformer encoder and NOT the full encoder-decoder Transformer (like translation models)?** Because you're **classifying**, not **generating** text. You need to *understand* the tweet and output one label — that's an encoder's job. A decoder generates new text, which you don't need. Your README says exactly this, and it's a great point to defend.

> **📝 Note (design coupling worth knowing):** step 6 *assumes* position 0 is the `<cls>` token. That's true **only because** preprocessing always prepends it (`add_cls=True`). The model and the preprocessing have a silent handshake. It's correct as written — just be aware that turning off `add_cls` would quietly break the logic. A common alternative to the `<cls>` trick is **mean-pooling** (average all token vectors instead of reading one slot).

**What else you could use instead of building this yourself:** download a **pretrained BERT** from Hugging Face and fine-tune it — far more accurate, but a black box. Building your own (like you did) is the right call for *learning how it works* and for an oral defense.

---

## 8. `train.py` — teaching the brain

**Its job:** the actual learning loop. Three functions.

### `train_one_epoch` — one full pass, with learning

For each batch of 32 tweets:
```python
optimizer.zero_grad()              # 1) clear last step's gradient
logits = model(input_ids)          # 2) predict
loss = criterion(logits, labels)   # 3) measure how wrong
loss.backward()                    # 4) compute the nudge direction
clip_grad_norm_(..., grad_clip)    # 5) cap the nudge size
optimizer.step()                   # 6) actually nudge the weights
```

This six-line ritual **is** deep learning. Memorize it — every PyTorch training loop on earth looks like this:
1. **zero_grad** — gradients pile up by default, so wipe them first.
2. **forward** — make predictions.
3. **loss** — score the wrongness (here, `CrossEntropyLoss`, the standard for classification).
4. **backward** — PyTorch computes, for every weight, which way to nudge it.
5. **grad clip** — if a nudge is dangerously huge, shrink it (`grad_clip=1.0`). Prevents training from blowing up.
6. **step** — the optimizer applies the nudges.

It also tracks **accuracy** and **macro-F1** as it goes (more on F1 below).

### `evaluate_one_epoch` — one pass, NO learning

Nearly identical, but:
- `model.eval()` and `torch.no_grad()` turn **off** dropout and gradient tracking. You're *grading*, not teaching, so the model must behave deterministically and you don't want to waste memory computing nudges.
- No `loss.backward()` / `optimizer.step()` — nothing learns here.

This runs on the **validation** set to check progress honestly.

### `fit_model` — the conductor of training

Loops over all 25 epochs. Each epoch: train once, validate once, record six numbers (train/val × loss/accuracy/F1), print a progress line, and:

```python
if val_f1 > best_val_macro_f1:
    best_val_macro_f1 = val_f1
    torch.save(model.state_dict(), best_model_path)   # save the best so far
```

**The key decision here:** it saves the model **whenever validation macro-F1 hits a new best** — not just the last epoch. So even if the model gets worse later (overfits), you keep the best version. `state_dict()` is just the bag of learned numbers.

**Why macro-F1 and not accuracy?** Your data is **imbalanced** — most airline tweets are negative. A lazy model could score high accuracy by *always* guessing "negative." **Macro-F1 averages the score across all three classes equally**, so the model is rewarded for getting neutral and positive right too. Saving the best model on macro-F1 is exactly the right choice for this dataset. Strong defense point.

> **📝 Note (what you'd add next):** there's no **early stopping** — it always runs all 25 epochs even if it stopped improving at epoch 10. Adding a "stop if no improvement for N epochs" is the natural next feature.

---

## 9. `evaluate.py` — the final exam

**Its job:** after training, grade the best model **once** on the **test** set — tweets it has never seen, not even during validation. This is the honest, final number.

- `predict(...)` runs the model over the test set and collects true vs. predicted labels.
- `compute_average_loss(...)` computes the test loss.
- `evaluate_on_test_set(...)` produces the full report: accuracy, macro precision/recall/F1, a **confusion matrix**, and a text report. It returns everything in one tidy dictionary.
- `save_test_report(...)` writes it to `outputs/reports/test_metrics.json`.

**Confusion matrix** = a grid showing, for each true class, what the model guessed. The diagonal is correct; off-diagonal cells show *which* mistakes (e.g. "neutral predicted as negative") — far more informative than a single accuracy number.

> **📝 Note:** `predict` and `compute_average_loss` each loop over the test set **separately** — two passes where one would do. Harmless at test time (it's small), but it's the kind of duplication you'd merge in a cleanup.

---

## 10. `plots.py` — the pictures

**Its job:** turn the recorded numbers into PNG charts using **matplotlib**.

- `plot_training_history(...)` → two line charts: **loss** over epochs and **accuracy** over epochs (train vs validation). The classic shape to look for: if validation loss starts rising while training loss keeps falling, that's **overfitting**.
- `plot_confusion_matrix(...)` → draws the confusion grid as a colored heatmap with the counts written in each cell.

> **📝 Note:** you *track* macro-F1 every epoch (it's your most important metric!) but you only **plot** loss and accuracy. Adding an F1 curve would be a nice, easy improvement.

---

## 11. `utils.py` — the toolbox

Small shared helpers used everywhere.

- **`set_seed(seed)`** — pins down all the random number generators (Python, NumPy, PyTorch, GPU). **Why it matters:** training involves randomness (shuffling, dropout, initial weights). Fixing the seed makes your results **reproducible** — run it twice, get the same numbers. Essential for a project you have to defend.
- **`create_directories(config)`** — makes the `data/` and `outputs/` folders if they don't exist, so saving never fails.
- **`save_json(obj, path)`** — saves a dictionary to a `.json` text file.
- **`compute_class_weights(labels, num_classes)`** — computes how much to "upweight" the rarer classes. The formula `total / (num_classes * count)` gives **bigger weights to rarer classes**, so the model is penalized more for getting them wrong. This is what `use_weighted_loss=True` switches on. It's the standard inverse-frequency balancing.

---

## 12. `main.py` — the conductor

This ties everything together for **training**. In order:

1. Load config, make folders, set the seed.
2. **Download + load** the tweets (`load_data.py`).
3. **Encode the labels** — `LabelEncoder` turns the words `negative/neutral/positive` into `0/1/2` (alphabetical order, so `negative=0, neutral=1, positive=2`). It saves this mapping so predictions can be turned back into words later.
4. **Split the data three ways** — test first (15%), then validation out of what's left, giving roughly **70% train / 15% val / 15% test**. `stratify=labels` keeps the class proportions identical in every split — important with imbalanced data.
5. **Clean + tokenize** all three splits.
6. **Build the vocabulary on the TRAINING set only** — `vocab.build(train_tokens)`. **Why only train?** If you built it from validation/test too, the model would get a peek at words it's supposed to be tested on cold — that's **data leakage**, a cardinal sin. Building on train only is correct and another strong defense point.
7. Convert all splits to padded ID sequences, wrap in `Dataset`s, wrap those in `DataLoader`s (which batch and shuffle).
8. Pick the **device** — GPU (`cuda`) if available, else CPU.
9. **Build the model** with sizes from config.
10. Build the **loss** (weighted, using `compute_class_weights`) and the **optimizer** (Adam).
11. **Train** with `fit_model`.
12. **Reload the best saved model** and run the **final test evaluation**.
13. Save the report and draw the plots.

This file is the story of your whole project in one place — if you can narrate `main.py` top to bottom, you understand the project.

---

## 13. `predict.py` — using the finished model

**Its job:** the "demo." Load the trained model and classify **one new tweet** you type in.

The flow mirrors training, but for a single tweet:
1. Load the saved `vocab.json`, `class_mapping.json`, and `best_model.pt`.
2. `encode_new_tweet(...)` runs the **same** clean → tokenize → add `<cls>` → pad steps on your tweet.
3. Run the model, apply **softmax** (turns the three raw scores into probabilities that add to 100%), and report the winning label plus a **confidence** %.

> **📝 Note (the biggest cleanup target):** `encode_new_tweet` **re-implements** the padding/truncation/`<cls>` logic that already exists in `preprocess.py` (`texts_to_padded_ids`). That's the same logic written twice. The risk: if you ever change how preprocessing works, you must remember to change it in *both* places, or training and prediction will silently disagree. The fix (later) is to call the existing `preprocess.py` functions here instead.

---

## 14. `src/__init__.py` — why an empty file exists

It's basically empty, and that's correct. Its mere presence tells Python "the `src` folder is a **package**," which is what lets `main.py` write `from src.model import ...`. Don't delete it.

---

## 15. The decisions that define your project (your defense cheat-sheet)

If you can answer these, you own this project:

| Question | Your answer |
|---|---|
| Where's the data from? | **Kaggle**, `crowdflower/twitter-airline-sentiment` (the README wrongly says OpenML — that's a doc bug). |
| Why a Transformer **encoder**, no decoder? | It's **classification**, not text generation. Encoders understand; decoders generate. |
| What's the `<cls>` token for? | A summary slot at position 0; after the encoder it represents the whole tweet, and the classifier reads only it. |
| Why measure **macro-F1**, not accuracy? | The data is **imbalanced** (mostly negative). Macro-F1 weights all classes equally so the model can't cheat by always saying "negative." |
| Why build the vocabulary on **train only**? | To avoid **data leakage** — the model must not peek at validation/test words. |
| Why **positional encoding**? | Transformers see all words at once and have no built-in sense of order; this injects it. |
| Why **weighted loss**? | To penalize mistakes on rare classes more, again because of imbalance. |
| How do you prevent **overfitting**? | Dropout (0.3), weight decay, and saving the best-validation model rather than the last. |
| Why **set a seed**? | Reproducibility — same results every run. |

---

## 16. Honest list of things you'd improve (none done yet)

You asked what *else* could go in. These are the real opportunities, in rough priority order:

1. **Fix the README** — it says OpenML; the code uses Kaggle.
2. **Remove the duplicate preprocessing in `predict.py`** — call `preprocess.py`'s functions instead (prevents train/predict drift).
3. **Merge the two CSV-finder functions** in `load_data.py`.
4. **Add early stopping** to `train.py` so it doesn't always run all 25 epochs.
5. **Plot the macro-F1 curve** (you already track it).
6. **Combine the two test-set passes** in `evaluate.py` into one.
7. (Bigger) Try a **subword tokenizer** or **pretrained embeddings** for higher accuracy.

None of these are bugs that stop it working — your pipeline is sound. They're polish.

---

## 17. How to actually run it

```bash
pip install -r requirements.txt   # install the libraries
python main.py                    # train from scratch (needs Kaggle access + internet first time)
python predict.py                 # classify a tweet you type in
```

The first `main.py` run downloads the data, trains for 25 epochs, and fills `outputs/` with the model, plots, and reports.

---

*That's the entire project. Read `main.py` alongside section 12, then `model.py` alongside section 7 — those two are the spine. Once they click, the rest is plumbing you already understand.*
