import time
import os

from src.utils import make_save_dir, read_json
from src.models import GenerationModel, TemplateHandler

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from transformers import T5Model, T5ForConditionalGeneration, T5Tokenizer
import logging

all_slots = [
'Brand Name', 
'Batteries Included?', 
'Product Dimensions', 
'Hardware Platform', 
'Hard Drive', 
'Max Screen Resolution', 
'Customer Reviews', 
'pid', 
'Processor Count', 
'Shipping Weight', 
'Battery Type', 
'Pattern', 
'Item Package Quantity', 
'Display Style', 
'Audio-out Ports (#)', 
'Item model number', 
'Shipping Information', 
'Graphics Coprocessor', 
'Hard Drive Rotational Speed', 
'Date First Available', 
'Processor', 
'Average Battery Life (in hours)', 
'Series', 
'Part Number', 
'title', 
'Optical Drive Type', 
'Item Dimensions L x W x H', 
'Power Source', 
'Warranty Description', 
'Best Sellers Rank', 
'RAM', 
'Number of USB 3.0 Ports', 
'Batteries', 
'International Shipping', 
'Domestic Shipping', 
'Flash Memory Size', 
'Manufacturer Part Number', 
'Graphics Card Ram Size', 
'Chipset Brand', 
'Screen Resolution', 
'Number of USB 2.0 Ports', 
'Voltage', 
'Rear Webcam Resolution', 
'Color Name', 
'Memory Speed', 
'Date first available at Amazon.com', 
'National Stock Number', 
'Card Description', 
'Wattage', 
'Special Features', 
'Batteries Required?', 
'Item Weight', 
'Usage', 
'category', 
'Screen Size', 
'Computer Memory Type', 
'ASIN', 
'Operating System', 
'Color', 
'Hard Drive Interface', 
'Wireless Type', 
'Processor Brand'
]


class Generator(object):
    """docstring for Generator"""
    def __init__(self, condition_generation=True, checkpoint='checkpoint.pth'):
        super(Generator, self).__init__()
        print('preparing dictionary...')
        # prepare dictionary, tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base', padding_side='right')
        self.word2id = self.tokenizer.get_vocab()
        self.bos = self.word2id[self.tokenizer.pad_token]

        self.src_max_len = 512
        self.tgt_max_len = 200
        print('preparing model...')
        # prepare model
        self._saved_checkpoint = checkpoint
        self.prepare_model(condition_generation=condition_generation)
        self._condition_generation = condition_generation
        

    def prepare_model(self, condition_generation=False):
        if condition_generation:
            self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        else:
            t5_model = T5Model.from_pretrained('t5-base')
            self.model = GenerationModel(t5_model)
        self.load_checkpoint()

    def _generate_one_step(self, input_ids):
        if self._condition_generation:
            outputs = self.model.generate(input_ids, max_length=self.tgt_max_len)
        else:
            outputs = self.model.generate(input_ids, max_length=self.tgt_max_len, bos_token=self.bos)
        return outputs

    def load_checkpoint(self):
        print('loading checkpoint')
        state_dict = torch.load(self._saved_checkpoint, map_location=torch.device('cpu'))['state_dict']
        self.model.load_state_dict(state_dict)

    def tokenize(self, x):
        return self.tokenizer.encode(x, max_length=self.src_max_len,
                                        truncation=True,
                                        return_tensors='pt',
                                        pad_to_max_length=True
                                    )

    def id2sent(self, indices):
        return ' '.join([self.index2word[int(w)] for w in indices])

    @torch.no_grad()
    def test(self, table):
        print('='*30)
        print('========== Testing ==========')
        print('='*30)

        self.model.eval()

        input_str = ''.join(['<'+k+'> '+v+' </'+k+'>\n' for k, v in table.items()])
        input_ids = self.tokenize(input_str).squeeze(0)
        print('input_ids', input_ids.size())
        outputs = self._generate_one_step(input_ids.unsqueeze(0))
        print('outputs', outputs.size())

        l = self.tokenizer.decode(outputs.squeeze(0).long().tolist())
        return l





