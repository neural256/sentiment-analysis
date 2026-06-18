import re
from collections import Counter
from typing import Iterable, List


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
URL_TOKEN = "<url>"
USER_TOKEN = "<user>"
CLS_TOKEN = "<cls>"


def clean_text(text: str) -> str:
    text = str(text).lower()

    text = re.sub(r"http\S+|www\.\S+", f" {URL_TOKEN} ", text)
    text = re.sub(r"@\w+", f" {USER_TOKEN} ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> List[str]:
    pattern = rf"{URL_TOKEN}|{USER_TOKEN}|[a-z]+(?:'[a-z]+)?|\d+"
    return re.findall(pattern, text)


class Vocabulary:
    def __init__(self, max_size: int = 10000, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq

        self.token_to_idx = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            URL_TOKEN: 2,
            USER_TOKEN: 3,
            CLS_TOKEN: 4,
        }

        self.idx_to_token = {
            0: PAD_TOKEN,
            1: UNK_TOKEN,
            2: URL_TOKEN,
            3: USER_TOKEN,
            4: CLS_TOKEN,
        }

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token_to_idx[UNK_TOKEN]

    @property
    def cls_idx(self) -> int:
        return self.token_to_idx[CLS_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def build(self, tokenized_texts: Iterable[List[str]]) -> None:
        counter = Counter()

        for tokens in tokenized_texts:
            counter.update(tokens)

        for token, freq in counter.most_common():
            if freq < self.min_freq:
                continue

            if token in self.token_to_idx:
                continue

            if len(self.token_to_idx) >= self.max_size:
                break

            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def encode_tokens(self, tokens: List[str]) -> List[int]:
        return [
            self.token_to_idx.get(token, self.unk_idx)
            for token in tokens
        ]


def pad_or_truncate(ids: List[int], max_length: int, pad_idx: int = 0) -> List[int]:
    if len(ids) > max_length:
        return ids[:max_length]

    return ids + [pad_idx] * (max_length - len(ids))


def texts_to_padded_ids(
    tokenized_texts: Iterable[List[str]],
    vocab: Vocabulary,
    max_length: int,
    add_cls: bool = True,
) -> List[List[int]]:
    all_ids = []

    for tokens in tokenized_texts:
        ids = vocab.encode_tokens(tokens)

        if add_cls:
            ids = [vocab.cls_idx] + ids

        ids = pad_or_truncate(ids, max_length=max_length, pad_idx=vocab.pad_idx)
        all_ids.append(ids)

    return all_ids
