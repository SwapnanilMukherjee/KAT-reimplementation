import torch
import random
import json
import numpy as np
import pickle
import os
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 hypothesis_prefix='hypothesis:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.hypothesis_prefix = hypothesis_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'gold_label' in example:
            return example['gold_label'] + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        hypothesis = self.hypothesis_prefix + " " + example['hypothesis']
        target = self.get_target(example)

        if 'entities' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            entities = example['entities'][:self.n_context]
            
            # Pad entities if less than n_context
            while len(entities) < self.n_context:
                entities = entities + entities[:(self.n_context-len(entities))]
            try:
                passages = [f.format(c[0], '{} is a {}'.format(c[0],c[1])) for c in entities]
            except Exception as e:
                print(example)
                print(entities)
                assert False
        else:
            passages = None

        return {
            'id': example['id'],
            'index': index,
            'hypothesis': hypothesis,
            'target': target,
            'passages': passages,
        }

    def get_example(self, index):
        return self.data[index]
    

