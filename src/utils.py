import json
import os
###### Some Utility Functions ######

def read_json(filename):
    '''Read in a json file.'''
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def make_save_dir(save_dir):
    '''Make a directory if it does not exist.'''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def cc(arr):
    '''Convert a function to cuda tensor'''
    return torch.from_numpy(np.array(arr)).cuda()


def tens2np(tensor):
    '''Convert tensor to numpy array'''
    return tensor.detach().cpu().numpy()

###### Some Utility Functions ######

