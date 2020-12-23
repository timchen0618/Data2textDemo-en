import os
import sys
import re
import json
import nltk.data
import nltk
from nltk.metrics.distance import edit_distance

def read_json(data):
    return json.load(open(data))

def split_sent(l):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')    
    return sent_detector.tokenize(l.strip('\n').strip())

def match(slots, sents):

    template_per_instance = []
    matches = []
    for sent in sents:
        sent_copy = nltk.word_tokenize(sent.strip('\n').strip())
        raw_template = sent
        # print('raw', raw_template)
        matched_slots = []
        matched_string = []
        m = 0
        for slot_name, value in slots.items():
            if value not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']:
                n_grams = [len(value.split(' '))]
                for n_gram in n_grams:
                    start = 0
                    end = n_gram
                    while 1:
                        if end > len(sent_copy):
                            break
                        cand = [l.lower() for l in sent_copy[start:end]]
                        # matching 
                        if cand == value.lower().split(' '):
                            m+=1
                            matched_string.append(sent_copy[start:end])
                            sent_copy[start:end] = ('<'+slot_name+'>').split(' ')
                            matched_slots.append(slot_name)

                        start += 1
                        end += 1

                # handle item model number problem
                if slot_name == 'Item model number' and len(value) >= 5:

                    for n_gram in range(1, 5):
                        start = 0
                        end = n_gram
                        while 1:
                            if end > len(sent_copy):
                                break
                            cand = ' '.join([l.lower() for l in sent_copy[start:end]])

                            # matching 
                            if float(edit_distance(cand, value.lower())) < 0.3 * len(value):
                                # see_file.write('[Model_Num] ')
                                # see_file.write(cand)
                                # see_file.write(' | ')
                                # see_file.write(value)
                                # see_file.write('\n')
                                m += 1
                                matched_string.append(sent_copy[start:end])
                                sent_copy[start:end] = ('<'+slot_name+'>').split(' ')
                                matched_slots.append(slot_name)
                                # see_item += 1

                            start += 1
                            end += 1

                if (slot_name in ['Processor', 'RAM', 'Hard Drive', 'Operating System', 'Computer Memory Type']) and len(value) >= 5:

                    for n_gram in range(1, 5):
                        start = 0
                        end = n_gram
                        while 1:
                            if end > len(sent_copy):
                                break
                            cand = ' '.join([l.lower() for l in sent_copy[start:end]])

                            # matching 
                            if float(edit_distance(cand, value.lower())) < 0.3 * len(value):
                                # see_file.write(cand)
                                # see_file.write(' | ')
                                # see_file.write(value)
                                # see_file.write('\n')
                                m += 1
                                matched_string.append(sent_copy[start:end])
                                sent_copy[start:end] = ('<'+slot_name+'>').split(' ')
                                matched_slots.append(slot_name)
                                # see_item += 1

                            start += 1
                            end += 1
                ##################################################
                ##################### TODO #######################
                ##################################################
                # handle partial match

                

        matches.append(m)
        sent_copy = ' '.join(sent_copy)
        start, end = sent_copy.find('<'), sent_copy.find('>')
        if matched_slots:
            template_per_instance.append((sent_copy, matched_slots, raw_template, matched_string))

    return template_per_instance
    

def highlight(sents, matched_string):
    # N sentences
    # N lists of True or False

    matches_or_not = []
    segments = []
    segments_hilight = []
    for sent in sents:
        
        match = [False for _ in sent]

        words = nltk.word_tokenize(sent.strip('\n').strip())

        for string in matched_string:
            start = sent.find(string)
            if start != -1:
                end = start + len(string)
                match[start:end] = [True for _ in range(end-start)]
        prev = match[0]
        start = 0
        for i in range(1, len(match)):
            if match[i] != prev:
                segments.append(sent[start:i])
                segments_hilight.append(prev)
                start = i
        segments.append(sent[start:])
        segments_hilight.append(match[-1])
    print('segments', segments)
    print('segments_hilight', segments_hilight)
    return segments, segments_hilight
    # return matches_or_not

if __name__ == '__main__':
    from tqdm import trange
    text = [l.strip('\n') for l in open(sys.argv[1])]
    data = read_json(sys.argv[2])
    for i in trange(len(text)):
        sents = split_sent(text[i])
        c = []
        a = match(data[i], sents)
        for mmm in a:
            c += [l for b in mmm[3] for l in b]
        assert False
