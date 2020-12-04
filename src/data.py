import time
import os

from torch.utils.data.dataset import Dataset
from transformers import T5Tokenizer
import torch

from utils import read_json


# class DataUtils(object):
#     """docstring for DataUtils"""
#     def __init__(self, args):
#         super(DataUtils, self).__init__()
#         self.args = args
#         self._train = args.train
#         self._data_path = args.data_path
#         if self._train:
#             self.table = read_json(os.path.join(self._data_path, args.table))
#             self.valid_table = read_json(os.path.join(self._data_path, args.valid_table))
#             self.descriptions = open(os.path.join(self._data_path, args.descriptions), 'r').readlines()
#             self.valid_descriptions = open(os.path.join(self._data_path, args.valid_descriptions), 'r').readlines()
#         else:
#             self.test_table = read_json(os.path.join(self._data_path, args.test_table))
#             self.test_descriptions = open(os.path.join(self._data_path, args.test_descriptions), 'r').readlines()


#     def read(filename):
#         def read_json(filename):
#             '''Read in a json file.'''
#             data = []
#             with open(filename) as json_file:
#                 for l in json_file:
#                     data.append(json.loads(l))

#             return data
#         data = read_json(filename)
#         descriptions = []
#         for l in data:
#             descriptions.append(l['description'])
#             del l['description']
        
#         def split_sent(data):
#             sent_data = []
#             sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
#             for l in data:
#                 sent_data.append(sent_detector.tokenize(l.strip('\n').strip()))

#             return sent_data

#         return data, split_sent(descriptions)

    


class ProductDataset(Dataset):
    """docstring for ProductDataset"""
    def __init__(self, args, valid, tokenizer):
        super(ProductDataset, self).__init__()
        self.args = args
        self._train = args.train
        self._valid = valid
        self._data_path = args.data_path
        if self._train:
            if valid:
                self.table = read_json(os.path.join(self._data_path, args.valid_table))
                self.descriptions = open(os.path.join(self._data_path, args.valid_descriptions), 'r').readlines()
            else:
                self.table = read_json(os.path.join(self._data_path, args.table))
                self.descriptions = open(os.path.join(self._data_path, args.descriptions), 'r').readlines()
        else:
            self.table = read_json(os.path.join(self._data_path, args.test_table))
            self.descriptions = open(os.path.join(self._data_path, args.test_descriptions), 'r').readlines()
            # self.table = read_json(os.path.join(self._data_path, args.table))
            # self.descriptions = open(os.path.join(self._data_path, args.descriptions), 'r').readlines()

        self.src_max_len = args.src_max_len
        self.tgt_max_len = args.tgt_max_len
        self.set_tokenizer(tokenizer)

    def set_tokenizer(self, tokenizer):
        # self.tokenizer = T5Tokenizer.from_pretrained('t5-base', padding_side='right')
        self.tokenizer = tokenizer
        self.word2id = self.tokenizer.get_vocab()
        # print(self.word2id)

        self.index2word = [[]]*len(self.word2id)
        for word in self.word2id:
            self.index2word[self.word2id[word]] = word

        self.vocab_size = len(self.word2id)
        print('vocab_size:',self.vocab_size)
        assert len(self.tokenizer) == self.vocab_size
        self.bos = self.word2id[self.tokenizer.pad_token]
        self.eos = self.word2id[self.tokenizer.eos_token]
        self.pad = self.word2id[self.tokenizer.pad_token]
        print('bos', self.bos)
        print('eos', self.eos)
        print('pad', self.pad)
        # self.eos = self.word2id['</s>']
        # self.bos = self.word2id['<s>']
        # self.pad = self.word2id['<pad>']
        # print('pad: ', self.pad)
        # print(self.tokenizer.get_vocab())

    def tokenize(self, x, is_tgt):
        if is_tgt:
            max_len = self.tgt_max_len
        else:
            max_len = self.src_max_len
        return self.tokenizer.encode(x, max_length=max_len,
                                        truncation=True,
                                        return_tensors='pt',
                                        pad_to_max_length=True
                                    )

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.bos
        pad_token_id = self.pad

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `labels` has only positive values and -100"

        return shifted_input_ids


    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        input_str = ''.join(['<'+k+'> '+v+' </'+k+'>\n' for k, v in self.table[index].items()])

        tgt_str = self.descriptions[index]

        input_ids = self.tokenize(input_str, is_tgt=False).squeeze(0)
        tgt_ids = self.tokenize(tgt_str + ' </s>', is_tgt=True).squeeze(0)
        # print(tgt_ids)

        # return {'input_ids': input_ids, 'input_str': input_str} 
        if self.args.template_decoding:
            return {'input_ids': input_ids, 'tgt_ids': tgt_ids, 'decoder_input_ids':self._shift_right(tgt_ids), 'table':self.table[index]} 
        else:
            return {'input_ids': input_ids, 'tgt_ids': tgt_ids, 'decoder_input_ids':self._shift_right(tgt_ids)} 
        # return {'input_ids': input_ids, 'decoder_input_ids': tgt_ids} 

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.descriptions)

    def id2sent(self, indices):
        return ' '.join([self.index2word[int(w)] for w in indices])

    def generate_target_file(self):
        target_file = open('target.txt', 'w')
        for l in self.descriptions:
            target_file.write(l)
        # print(len(self.descriptions))



