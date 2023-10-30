import pandas as pd
import torch
from datasets import Dataset

class GetSentence(object):
    def __init__(self,data):
        self.data = data
        self.n_sentences = 1
        self.empty = False
        agg_function = lambda s: [(w,t) for w,t in zip(s["Word"].values.tolist(),
                                                        s["Tag"].values.tolist())]
        self.group = self.data.groupby('Sentence #').apply(agg_function)
        self.sentence = [s for s in self.group]


class NerDataset(Dataset):
    def __init__(self, file_name):
        data = pd.read_csv(file_name, encoding='latin1')

        data = data.fillna(method='ffill')

        getter = GetSentence(data[:100])

        self.x = [" ".join([word[0] for word in sentence]) for sentence in getter.sentence]
        self.y = [" ".join([lab[1] for lab in sentence]) for sentence in getter.sentence]


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]