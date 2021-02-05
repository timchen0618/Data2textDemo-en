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
    '''
        args:
            slots: all attributes in the table
            sents: descriptions
    '''
    template_per_instance = []
    matches = []
    for sent in sents:
        sent_copy = nltk.word_tokenize(sent.strip('\n').strip())
        raw_template = sent
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
                                m += 1
                                matched_string.append(sent_copy[start:end])
                                sent_copy[start:end] = ('<'+slot_name+'>').split(' ')
                                matched_slots.append(slot_name)

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
    '''
        template_per_instance, each element contains:
            1. original sentence
            2. matched_slots (should be highlighted)
            3. matched template
            4. matched string in the description (should be highlighted)
    '''
    return template_per_instance
    

def highlight(sents, matched_string):
    # N sentences
    # N lists of True or False
    '''
        split the whole description into many segments, and identify which segments should be highlighted
        segments: word sequence segments
        segments_hilight: bool, whether highlight (length equals to segments)
    '''
    print('matchcccccc', matched_string)
    segments = []
    segments_hilight = []
    for sent in sents:
        
        match = [False for _ in sent]  # whether each character in the sentence is a match

        words = nltk.word_tokenize(sent.strip('\n').strip())

        '''
            for every matched string, find it in the sentence and label corresponding character as match
            e.g. 
                sent: 'The 27-inch iMac with Retina 5K display' 
                matched_string: ['iMac']
                match: [F F F F F F F F F F F F T T T T F F F F ....] (F -> False, T -> True)
        '''
        for string in matched_string:
            string = ' '.join(string)  # concatenate the matched words of the same instance into a single string
            print('string', string)
            start = sent.find(string)  
            if start != -1:            # if find some matched string in this sentence
                print('string', string)
                print('start', start)
                end = start + len(string)
                print('end', end)
                
                for i in range(start, end):
                    match[i] = True
        
        prev = match[0]
        print('match', match)
        
        start = 0
        for i in range(1, len(match)):
            # if encounter switch in boolearn value e.g. F->T (means the border of segment)
            if match[i] != prev and sent[i] != ' ': # not label ' ' as highlight
                print('start', start)
                print('i', i)
                if sent[start:i] != ' ':
                    segments.append(sent[start:i].strip())  # append segments into a list 
                    print('sent', sent[start:i].strip())
                    segments_hilight.append(prev)           # append corresponding boolean value of the segments
                start = i
                prev = match[i]

        # append the last segment
        segments.append(sent[start:])      
        segments_hilight.append(match[-1])
    print('segments', segments)
    print('segments_hilight', segments_hilight)
    print('matched_string', matched_string)
    return segments, segments_hilight

if __name__ == '__main__':
    from tqdm import trange
    text = [l.strip('\n') for l in open(sys.argv[1])]
    data = read_json(sys.argv[2])
    for i in trange(len(text)):
        sents = split_sent(text[i])
        c = []
        a = match(data[i], sents)
        for mmm in a:
            c += [b for b in mmm[3]]
        segments, match_or_not_v2 = highlight(sents, c)
        assert False
