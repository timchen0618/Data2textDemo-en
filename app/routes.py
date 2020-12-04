from app import app
from flask import render_template, flash, request, redirect, url_for
from app.forms import LoginForm
from src.test import Generator, all_slots
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
        tuples = list(table.items())
        return render_template('index.html', form=form, table=tuples, description=description, product_list=NAMES)

    tuples = list(table.items())
    return render_template('index.html', form=form, table=tuples, description=description, product_list=NAMES)

@app.route('/attr', methods=['GET', 'POST'])
def attr():
    return render_template('attr.html', slots=all_slots)


