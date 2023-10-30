from collections import Iterable

import pandas as pd
from datasets import load_dataset
from pandas import DataFrame
from sklearn.metrics import classification_report, f1_score
from transformers import BertTokenizerFast, Trainer, TrainingArguments, default_data_collator, \
    DataCollatorForTokenClassification, DataCollator
from transformers.trainer_utils import IntervalStrategy
import torch

from models import BertCRF
from sklearn.metrics import precision_recall_fscore_support

from util import NerDataset

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
from sklearn.model_selection import train_test_split

MAX_LEN = 75
BATCH_SIZE = 16


model = BertCRF.from_pretrained('bert-base-uncased', num_labels=18).to(device)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

ner_tag = {'I-tim': 0,'B-org': 1,'I-org': 2,'B-gpe': 3,'I-art': 4,'B-per': 5,
'O': 6,'I-per': 7,'I-gpe': 8,'I-nat': 9,'B-nat': 10,'B-geo': 11,'B-eve': 12,'I-eve': 13,'B-tim': 14,'I-geo': 15,'B-art': 16,'PAD': 17}

class GetSentence(object):
    def __init__(self,data):
        self.data = data
        self.empty = False
        sentences = data['Sentence #'].values
        data['Sentence #'] = self._fill_nan(sentences)
        agg_function = lambda s: [(w,t) for w,t in zip(s["Word"].values.tolist(),
                                                        s["Tag"].values.tolist())]
        self.group = self.data.groupby('Sentence #').apply(agg_function)
        self.sentence = [s for s in self.group]

    def _fill_nan(self,sentences):
        current_sentence = sentences[0]
        for i in range(1,len(sentences)):
            if pd.isnull(sentences[i]):
                sentences[i] = current_sentence
            else:
                current_sentence = sentences[i]
        return sentences

def tokenize_function(examples):
    result = {
        'labels': [],
        'input_ids': [],
        'token_type_ids': []
    }

    getter = GetSentence(DataFrame(dict(examples)))

    sentences = [" ".join([word[0] for word in sentence]) for sentence in getter.sentence]
    labels = [[lab[1] for lab in sentence] for sentence in getter.sentence]

    max_length = tokenizer.max_model_input_sizes['bert-base-cased']

    tokenids = tokenizer(sentences, add_special_tokens=False)
    for ids,lab in zip(tokenids['input_ids'],labels):

        labels= [ner_tag[item] for item in lab]
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(ids)
        token_ids = tokenizer.build_inputs_with_special_tokens(ids)
        labels.insert(0, 0)
        labels.insert(0, 17)
        result['input_ids'].append(token_ids)
        result['labels'].append(labels)
        result['token_type_ids'].append(token_type_ids)
    result = tokenizer.pad(result, padding='longest', max_length=max_length, return_attention_mask=True)


    for i in range(len(result['input_ids'])):
        diff = len(result['input_ids'][i]) - len(result['labels'][i])
        result['labels'][i] += [17] * diff
    return result

dataset = load_dataset('csv',cache_dir="",data_files={'train': 'ner_train.csv', 'test': 'ner_test.csv'},encoding="ISO-8859-1")

# 4194300
train_datasets = dataset['train'].map(tokenize_function,batch_size=1048577,remove_columns=['Sentence #', 'Word', 'POS', 'Tag'],batched=True)

test_datasets = dataset['test'].map(tokenize_function,batch_size=1048577,remove_columns=['Sentence #', 'Word', 'POS', 'Tag'],batched=True)

print(train_datasets.features)

print(train_datasets,test_datasets)

def compute_metrics(pred):
    labels = pred.labels.flatten()
    preds = pred.predictions.flatten()
    f1 = precision_recall_fscore_support(labels, preds, average='weighted')
    report = classification_report(labels, preds)
    print(report)
    print("f1",f1)
    return {
        'f1': f1
    }


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=200,
    weight_decay=0.01,
    save_strategy=IntervalStrategy.EPOCH,
    logging_dir='./logs',
)


trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_datasets,
    eval_dataset=test_datasets
)

trainer.train()

print(trainer.evaluate())
