from comet_ml import Experiment
import torch
import torch.nn as nn
from transformers import T5Model, T5ForConditionalGeneration, T5Tokenizer
from data import ProductDataset
from torch.utils.data import DataLoader
from utils import make_save_dir, read_json
import time
from models import GenerationModel, TemplateHandler
import os
import numpy as np

class Solver(object):
    """docstring for Solver"""
    def __init__(self, args):
        super(Solver, self).__init__()
        self.args = args
        self._print_every_step = args.print_every_step
        self._valid_every_step = args.valid_every_step
        self._save_checkpoints = args.save_checkpoints
        if args.train:
            self.model_dir = make_save_dir(os.path.join(args.model_path, args.exp_name))
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        self._disable_comet = args.disable_comet
        self._condition_generation = args.condition_generation
        self._saved_checkpoint = args.load_model

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base', padding_side='right')

        if args.test:
            self.outfile = open(os.path.join(args.pred_dir, args.prediction), 'w')
            self.template_decoding = args.template_decoding
            if args.template_decoding:
                all_slots = [l.strip('\n') for l in open(args.f_all_slots)]
                all_templates = [l.strip('\n') for l in open(args.f_all_templates)]
                self.temp = TemplateHandler(all_slots, all_templates, self.tokenizer)
                    
        self.prepare_model(condition_generation=args.condition_generation, template_decoding=args.template_decoding)

        
    def prepare_model(self, condition_generation=False, template_decoding=False):
        print('condition_generation: ', condition_generation)
        if condition_generation:
            self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        else:
            t5_model = T5Model.from_pretrained('t5-base')
            if template_decoding:
                self.model = GenerationModel(t5_model, self.temp)
            else:
                self.model = GenerationModel(t5_model)
        self.lr = 1e-3
        self.model = self.model.cuda()
        


    def loss_compute(self, out, target, padding_idx):
        true_dist = out.data.clone()
        true_dist.fill_(0.)
        true_dist.scatter_(2, target.unsqueeze(2), 1.)
        true_dist[:,:,padding_idx] *= 0
        return -(true_dist*out).sum(dim=2).mean()

    def cal_avg_len(self, train_loader):
        total = 0
        input_lens = []
        tgt_lens = []
        input_long = 0
        tgt_long = 0

        for batch in train_loader:
            input_lens.append(batch['input_lens'])
            tgt_lens.append(batch['tgt_lens'])
            if batch['input_lens'] > 512:
                input_long += 1
            if batch['tgt_lens'] > 256:
                tgt_long += 1

        print(float(sum(input_lens))/len(input_lens))
        print(float(sum(tgt_lens))/len(tgt_lens))
        print(input_long)
        print(tgt_long)

    def cal_number_of_templates(self):
        templates = [l.strip('\n').strip() for l in open('all_templates.txt', 'r').readlines()]
        matched_slots = read_json('matched_slots.json')
        train_set = ProductDataset(self.args, valid=False, tokenizer=self.tokenizer)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)

        num_templates_list = []
        step = 0
        for batch in train_loader:
            if step % 100 == 0:
                print(step)
            num_templates = 0
            # print('===========')
            # print(batch['table'])
            for i, t in enumerate(templates):
                viable = True
                for slot in matched_slots[i]:
                    # print(slot)
                    if slot not in batch['table']:
                        viable = False
                        break
                if viable:
                    num_templates += 1
            # print(num_templates)
            num_templates_list.append(num_templates)
            # time.sleep(1)
            step += 1

        print(float(sum(num_templates_list))/len(num_templates_list))


    def _convert_ids_to_string(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True) 

    def _run_one_step(self, batch):
        input_ids = batch['input_ids'].cuda()  # (batch, length (max 512))
        decoder_input_ids = batch['decoder_input_ids'].cuda()     # (batch, length (max 200))
        tgt_ids = batch['tgt_ids'].cuda()

        ## Forwarding ## 
        if self._condition_generation:
            out = self.model(input_ids=input_ids, labels=tgt_ids)
            loss, prediction_scores = out[:2]
            # len of out -> 4
        else:
            out = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            loss = self.loss_compute(out, tgt_ids, 0)
        loss.backward()
        return loss

    def _generate_one_step(self, batch):
        input_ids = batch['input_ids'].cuda()
        if self._condition_generation:
            outputs = self.model.generate(input_ids, max_length=self.args.tgt_max_len)
        else:
            if self.template_decoding:
                outputs = self.model.generate(input_ids, max_length=self.args.tgt_max_len, bos_token=self.test_set.bos, table=batch['table'])
            else:
                outputs = self.model.generate(input_ids, max_length=self.args.tgt_max_len, bos_token=self.test_set.bos)
        return outputs

    def id2sent(self, indices):
        return ' '.join([self.index2word[int(w)] for w in indices])

    def load_checkpoint(self):
        state_dict = torch.load(self._saved_checkpoint)['state_dict']
        self.model.load_state_dict(state_dict)

    def train(self):
        if not self._disable_comet:
            # logging
            COMET_PROJECT_NAME = 'data2text'
            COMET_WORKSPACE = 'timchen0618'

            self.exp = Experiment(project_name=COMET_PROJECT_NAME,
                                  workspace=COMET_WORKSPACE,
                                  auto_output_logging='simple',
                                  auto_metric_logging=None,
                                  display_summary=False,
                                 )
            self.exp.set_name(self.args.exp_name)

        optim = torch.optim.Adam(self.model.parameters(), lr = self.lr, betas=(0.9, 0.98), eps=1e-9)
        train_set = ProductDataset(self.args, valid=False, tokenizer=self.tokenizer)
        print('[Logging Info] Finish loading data, start training..., length: %d'%len(train_set))
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

        step = 0
        losses = []
        for epoch in range(self.num_epoch):
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

            for batch in train_loader:
                self.model.train()
                optim.zero_grad()
                loss = self._run_one_step(batch)
                optim.step()
                losses.append(loss)

                if step % self._print_every_step == 0:
                    print('Logging...')
                    print('Step: %d | Loss: %f'%(step, sum(losses)/len(losses)))
                    if not self._disable_comet:
                        self.exp.log_metric('Train Loss', sum(losses)/len(losses), step=step)
                    losses = []
                    if self._condition_generation:
                        pred = self._generate_one_step(batch)[0].squeeze(0)
                        print(self._convert_ids_to_string(pred))

                if step % self._valid_every_step == self._valid_every_step - 1:
                    self.validate(step)

                step += 1
                print('-'*64)


    def validate(self, step):
        print('='*33)
        print('========== Validation ==========')
        print('='*33)

        self.model.eval()
        valid_set = ProductDataset(self.args, valid=True, tokenizer=self.tokenizer)
        print('Valid Set Size: %d'%len(valid_set))
        valid_loader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=0)

        losses = []

        for batch in valid_loader:
            loss = self._run_one_step(batch)
            losses.append(loss)

        print('Valid Loss: %4.6f'%(sum(losses)/len(losses)))
        if not self._disable_comet:
            self.exp.log_metric('Valid Loss', sum(losses)/len(losses), step=step)

        if self._save_checkpoints:
            print('saving!!!!')

            model_name = str(int(step/1000)) + 'k_' + '%6.6f_'%(sum(losses)/len(losses)) + 'model.pth'
            state = {'step': step, 'state_dict': self.model.state_dict()}
            torch.save(state, os.path.join(self.model_dir, model_name))



    @torch.no_grad()
    def test(self):
        print('='*30)
        print('========== Testing ==========')
        print('='*30)

        self.load_checkpoint()
        self.model.eval()

        test_set = ProductDataset(self.args, tokenizer=self.tokenizer, valid=False)
        self.test_set = test_set
        print('Finish loading dataset, length: %d'%len(test_set))
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

        losses = []

        start_time = time.time()
        i = 0
        for batch in test_loader:
            outputs = self._generate_one_step(batch)
            # print('outputs', outputs.size())
            for l in outputs:
                if self._condition_generation:
                    l = self.tokenizer.decode(l.long().tolist())
                else:
                    l = self.tokenizer.decode(l.long().tolist())
                # print('l', l)
                self.outfile.write(l)
                self.outfile.write('\n')
            # self.outfile.write(batch['input_str'][0])
            # self.outfile.write('='*20)
            # self.outfile.write('\n')
            i += 1
            if i % 10 == 0:
                print('Step: %d | Time Elapsed: %f'%(i, time.time()-start_time))
                start_time = time.time()

        self.outfile.close()

    # def generate_target_file(self):
    #     test_set = ProductDataset(self.args, tokenizer=self.tokenizer, valid=False)
    #     print('Finish loading dataset, length: %d'%len(test_set))
    #     test_set.generate_target_file()

    

