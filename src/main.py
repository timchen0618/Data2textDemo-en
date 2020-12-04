import argparse
from solver import Solver


def parse():
    parser = argparse.ArgumentParser(description="tree transformer")

    parser.add_argument('-load', default=None, help= 'load: model_dir', dest= 'load_model', type=str)
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')

    parser.add_argument('-disable_comet', action='store_true')
    parser.add_argument('-print_every_step', type=int, default=1000)
    parser.add_argument('-valid_every_step', type=int, default=40000)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-num_epoch', type=int, default=20)
    parser.add_argument('-exp_name', type=str)
    parser.add_argument('-condition_generation', action='store_true')
    parser.add_argument('-template_decoding', action='store_true')

    parser.add_argument('-data_path', type=str, default='./processed_data/')
    parser.add_argument('-table', type=str, default='table.train.json')
    parser.add_argument('-descriptions', type=str, default='description.train.txt')
    parser.add_argument('-valid_table', type=str, default='table.valid.json')
    parser.add_argument('-valid_descriptions', type=str, default='description.valid.txt')
    parser.add_argument('-test_table', type=str, default='table.test.json')
    parser.add_argument('-test_descriptions', type=str, default='description.test.txt')

    # for template decode
    parser.add_argument('-f_all_slots', type=str, default='./processed_data/all_slots_brackets.txt')
    parser.add_argument('-f_all_templates', type=str, default='./templates/all_templates.txt')

    parser.add_argument('-src_max_len', type=int, default=512)
    parser.add_argument('-tgt_max_len', type=int, default=200)

    parser.add_argument('-model_path', type=str, default='train_model')
    parser.add_argument('-save_checkpoints', action='store_true')

    parser.add_argument('-pred_dir', type=str, default='./pred_dir/')
    parser.add_argument('-prediction', type=str, default='pred.txt')


    

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    print('[Logging Info] Finish building solver...')
    if args.train:
        solver.train()
    elif args.test:
        solver.test()
    # solver.cal_number_of_templates()










