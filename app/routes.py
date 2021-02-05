from app import app
from flask import render_template, flash, request, redirect, url_for
from app.forms import LoginForm
from src.test import Generator, all_slots
from src.match import match, split_sent, highlight
# from app.control import ctrl
import json

# some selected example table for user to click on 
TABLE = json.load(open('data/new_table.json'))
# names displayed on the website for the examples
NAMES = ['Dell Laptop Latitude E6440', 
         'Dell Vostro 3458', 
         'Apple iMac 27', 
         'HP Z820 Workstation', 
         'iBUYPOWER Gamer Supreme', 
         'Toshiba Tecra C50-B1503', 
         'Asus Z91', 
         'Lenovo T530'
         ]
# init generator
generator = Generator(checkpoint='src/checkpoint.pth')
print('finish loading generator')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = LoginForm() 
    product = request.args.get('product')  # the id of selected example product
    table = {}
    description = ""

    matched_slots = []
    sents = []
    # matched_sentences = []
    # match_or_not = []
    matched_string = []
    segment_highlight_or_not = []
    segments = []

    try:
        print('pro', product)
        num = int(product)                 # the id of selected example product -> if this line works, means it is int
                                           # means user have clicked on some example -> need to display the attr table
        # get the table and example name
        table = TABLE[num-1]
        name = NAMES[num-1]
        global_table = table
    except:
        if product == 'submit':
            ### handle table here ###
            num_form = (len(request.form))//2
            table = {}
            form_table = request.form.to_dict(flat=False)   # convert table into a dictionary

            # get the input table through a form (displayed as a modifiable table)
            # pass to python code using request.form
            for i in range(num_form):     # fill the attribute and value to table
                if not request.form['attr-%d'%i]:
                    continue
                table[request.form['attr-%d'%i]] = request.form['value-%d'%i]

            if table:
                ######################################
                # generate a description based on the table
                ######################################
                description = generator.test(table) 

                sents = split_sent(description)   # split the description into sentences
                matches = match(table, sents)
                for l in matches:
                    matched_slots += l[1]
                # matched_sentences = [l[2] for l in matches]
                
                for mmm in matches:
                    matched_string += [b for b in mmm[3]]  # mmm[3] -> matched string per sentence in description
                
                '''
                    split the whole description into many segments, and identify which segments should be highlighted
                    segments: word sequence segments
                    segments_hilight: bool, whether highlight (length equals to segments)
                '''
                segments, segment_highlight_or_not = highlight(sents, matched_string)


                # for l in sents:
                #     if l in matched_sentences:
                #         match_or_not.append(True)
                #     else:
                #         match_or_not.append(False)
                print(matched_slots)
                print('sssssss')
                # print(matched_sentences)
                print(sents)

        tuples = list(table.items())
        return render_template('index.html', form=form, table=tuples, description=segments, product_list=NAMES, matched_slots=matched_slots, match_or_not=segment_highlight_or_not)

    tuples = list(table.items())
    return render_template('index.html', form=form, table=tuples, description=segments, product_list=NAMES, matched_slots=matched_slots, match_or_not=segment_highlight_or_not)

# display all possible attributes
@app.route('/attr', methods=['GET', 'POST'])
def attr():
    return render_template('attr.html', slots=all_slots)


