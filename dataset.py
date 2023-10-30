import pandas as pd
import numpy as np
from typing import *
from numbers import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch import Tensor
from tqdm.auto import tqdm


class NERDataset(object):
    def __init__(self,filepath : str,
                 sentence_transform : Callable[Iterable[str],Iterable[str]] = None) -> None:
       # print(filepath)
        df = pd.read_csv(filepath,encoding="ISO-8859-1")
        sentences = df['Sentence #'].values
        df['Sentence #'] = self._fill_nan(sentences)
        df.columns = ["sentence","word","pos","tag"]
        
        self.collection = []
        for _,subdf in df.groupby(by="sentence"):
            self.collection.append(
                {
                    "words" : subdf['word'].values.tolist(),
                    "pos" : subdf['pos'].values.tolist(),
                    "tag" : subdf['tag'].values.tolist()
                }
            )
        self.sentence_transform = sentence_transform


    def _fill_nan(self,sentences : Iterable[str]) -> Iterable[str] :
        current_sentence = sentences[0]
        for i in range(1,len(sentences)):
            if pd.isnull(sentences[i]):
                sentences[i] = current_sentence
            else:
                current_sentence = sentences[i]
        return sentences
    
    def __getitem__(self,idx : Integral ) -> dict:
        data = self.collection[idx]
        if not self.sentence_transform is None:
            data['words'] = self.sentence_transform(data['words'])
        return data
    
    def __len__(self) -> Integral:
        return len(self.collection)



    
class TensorNerDataset(NERDataset,Dataset):
    def __init__(self,filepath : str,
                 sentence_transform : Callable[Iterable[str],Iterable[str]] = None,
                 max_lengths : Integral = 40,
                 padding_token = "<<EOF>>",
                 padding_tag   = "O") -> None:
        NERDataset.__init__(self,filepath=filepath,sentence_transform=sentence_transform)
        Dataset.__init__(self)
        self.word2idx = {padding_token : 0}
        self.tag2idx  = {padding_tag : 0}
        self.padding_token = padding_token
        self.padding_tag = padding_tag
        idx = 1
        pid = 1
        self.max_lengths = max_lengths
        for i in tqdm(range(len(self.collection))):
            if not self.sentence_transform is None:
                self.collection[i]['words'] = self.sentence_transform(self.collection[i]['words'])
            self.collection[i] = self.padding(self.collection[i])
            for word in self.collection[i]['words']:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    idx += 1
            for tag in self.collection[i]['tag']:
                if tag not in self.tag2idx:
                    self.tag2idx[tag] = pid
                    pid += 1
        self.idx2tag  = { idx : tag  for tag,idx in self.tag2idx.items() }
        self.idx2word = { idx : word for word,idx in self.word2idx.items() }
    
    def padding(self,data : dict) -> dict:
        for k in data:
            data[k] = data[k][:self.max_lengths]
            while len(data[k]) < self.max_lengths:
                if k == "words":
                    data[k].append(self.padding_token)
                elif k == "tag":
                    data[k].append(self.padding_tag)
                else:
                    data[k].append("None")
        return data
    def __getitem__(self,idx : Integral) -> Tuple[Tensor,Tensor]:
        data = self.collection[idx]
        words= [ self.word2idx[word] for word in data['words'] ] 
        pos  = [ self.tag2idx[pos] for pos in data['tag'] ]
        
        return torch.LongTensor(words),torch.LongTensor(pos)