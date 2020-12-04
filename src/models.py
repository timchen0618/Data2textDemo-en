import torch
import torch.nn as nn

class GenerationModel(torch.nn.Module):
    """docstring for GenerationModel"""
    def __init__(self, T5, temp=None):
        super(GenerationModel, self).__init__()
        self.t5 = T5
        size = tuple(self.t5.get_input_embeddings().weight.size())
        self.linear = nn.Linear(size[1], size[0])
        self.sm = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.temp = temp

    def forward(self, input_ids, decoder_input_ids):
        out = self.t5(input_ids=input_ids, decoder_input_ids=decoder_input_ids)[0]
        out = self.sm(self.linear(out))
        return out

    def generate(self, input_ids, max_length, bos_token, table=None):
        if self.temp:
            return self._template_decode(input_ids, max_length, bos_token, table)
        else:
            return self._generate(input_ids, max_length, bos_token)
        
    def _generate(self, input_ids, max_length, bos_token):
        # Encode if needed (training, first prediction pass)
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids
        )

        hidden_states = encoder_outputs[0] # (batch, length, dim)
        # print('hidden_states', hidden_states.size())
        b_size, length, dim = hidden_states.size()
        decoder_input_ids = torch.zeros((b_size, 1)).fill_(bos_token).long().cuda()

        for step in range(max_length):
            # Decode
            # print('input', decoder_input_ids.size())
            decoder_outputs = self.t5.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=hidden_states
            )
            decoder_out = self.softmax(self.linear(decoder_outputs[0][:, -1]))
            decoder_out = decoder_out.unsqueeze(1)
            # print(decoder_out.size())
            # print(decoder_out.argmax(dim=-1).size())
            decoder_input_ids = torch.cat((decoder_input_ids, decoder_out.argmax(dim=-1).long()), dim = 1)
        return decoder_input_ids[:, 1:]

    def _template_decode(self, input_ids, max_length, bos_token, table):
        # construct candidate set
        self.temp.set_candidate_sents(table)
        self.temp.set_candidate_toks()
        # print(self.temp.candid_toks)
        assert False
        # Encode if needed (training, first prediction pass)
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids
        )

        hidden_states = encoder_outputs[0] # (batch, length, dim)
        # print('hidden_states', hidden_states.size())
        b_size, length, dim = hidden_states.size()
        decoder_input_ids = torch.zeros((b_size, 1)).fill_(bos_token).long().cuda()

        for step in range(max_length):
            # Decode
            # print('input', decoder_input_ids.size())
            decoder_outputs = self.t5.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=hidden_states
            )
            decoder_out = self.softmax(self.linear(decoder_outputs[0][:, -1]))
            decoder_out = decoder_out.unsqueeze(1)
            # print(decoder_out.size())
            # print(decoder_out.argmax(dim=-1).size())
            decoder_input_ids = torch.cat((decoder_input_ids, decoder_out.argmax(dim=-1).long()), dim = 1)
        return decoder_input_ids[:, 1:]


class TemplateHandler(object):
    """docstring for TemplateHandler"""
    def __init__(self, all_slots, all_templates, tokenizer):
        super(TemplateHandler, self).__init__()
        self.all_slots = all_slots
        self.all_templates = all_templates
        self.tokenizer = tokenizer
        self.candid_sents = []
        self.candid_toks = []
        
    def fill_slots_with_values(self, sent, table):
        all_in, slots, out_sent = self.check_all_in_tablev2(sent, table)
        if not all_in:
            return ""
        else:
            return out_sent


    def check_all_in_tablev2(self, sent, table):
        slots = []
        for slot in self.all_slots:
            idx = sent.find(slot)
            if idx != -1:
                value = table.get(slot[1:-1])
                if value:
                    slots.append((slot, value))
                    sent = sent.replace(slot, value)
                else:
                    return False, [], ""
        return True, slots, sent

    def set_candidate_sents(self, table):
        self.candid_sents = []
        for i in range(len(self.all_templates)):
            sent = self.fill_slots_with_values(self.all_templates[i], table)
            if sent:
                self.candid_sents.append(sent)

        print('produce candidate of %d sentences'%(len(self.candid_sents)))


    def set_candidate_toks(self):
        def tokenize(self, x):
            return self.tokenizer.encode(x, padding='longest')

        for l in self.candid_sents:
            self.candid_toks.append(tokenize(l))

        for l in self.candid_toks:
            print(l)