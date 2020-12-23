from app import app
from flask import render_template, flash, request, redirect, url_for
from app.forms import LoginForm
from src.test import Generator, all_slots
from src.match import match, split_sent, highlight
# from app.control import ctrl
import json

TABLE = json.load(open('data/new_table.json'))

NAMES = ['Dell Laptop Latitude E6440', 
         'Dell Vostro 3458', 
         'Apple iMac 27', 
         'HP Z820 Workstation', 
         'iBUYPOWER Gamer Supreme', 
         'Toshiba Tecra C50-B1503', 
         'Asus Z91', 
         'Lenovo T530'
         ]
generator = Generator(checkpoint='src/checkpoint.pth')
print('finish loading generator')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = LoginForm()
    product = request.args.get('product')
    table = {}
    description = ""
    print('damn')
    matched_slots = []
    sents = []
    matched_sentences = []
    match_or_not = []
    matched_string = []
    match_or_not_v2 = []
    segments = []

    # if form.validate_on_submit():
    #     print('fff')
    #     return render_template('index.html', form=form, table=tuples, description=description, product_list=NAMES)

    try:
        print('pro', product)
        num = int(product)
        table = TABLE[num-1]
        name = NAMES[num-1]
        global_table = table
    except:
        if product == 'submit':
            print('ggg')
            ### handle table here ###
            
            # print(request.form)
            num_form = (len(request.form))//2
            table = {}
            form_table = request.form.to_dict(flat=False)
            # print(form_table)
            for i in range(num_form):
                # print(form_table['attr-%d'%i], form_table['value-%d'%i])
                if not request.form['attr-%d'%i]:
                    continue
                table[request.form['attr-%d'%i]] = request.form['value-%d'%i]

            if table:
                print('table', table)
                description = generator.test(table) 
                print(description)
                sents = split_sent(description)
                matches = match(table, sents)
                for l in matches:
                    matched_slots += l[1]
                matched_sentences = [l[2] for l in matches]
                
                for mmm in matches:
                    matched_string += [b for b in mmm[3]]
                segments, match_or_not_v2 = highlight(sents, matched_string)


                for l in sents:
                    if l in matched_sentences:
                        match_or_not.append(True)
                    else:
                        match_or_not.append(False)
                print(matched_slots)
                print('sssssss')
                print(matched_sentences)
                print(sents)

        tuples = list(table.items())
        return render_template('index.html', form=form, table=tuples, description=segments, product_list=NAMES, matched_slots=matched_slots, matched_sents=matched_sentences, match_or_not=match_or_not_v2)

    tuples = list(table.items())
    return render_template('index.html', form=form, table=tuples, description=segments, product_list=NAMES, matched_slots=matched_slots, matched_sents=matched_sentences, match_or_not=match_or_not_v2)

@app.route('/attr', methods=['GET', 'POST'])
def attr():
    return render_template('attr.html', slots=all_slots)


