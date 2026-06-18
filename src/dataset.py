import torch
from torch.utils.data import Dataset


class TweetSentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

        if len(self.sequences) != len(self.labels):
            raise ValueError("Number of sequences and labels must match.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
