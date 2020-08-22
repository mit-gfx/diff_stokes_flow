import numpy as np

def ndarray(val):
    return np.asarray(val, dtype=np.float64)

def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')

def print_ok(*message):
    print('\033[92m', *message, '\033[0m')

def print_warning(*message):
    print('\033[93m', *message, '\033[0m')

def print_info(*message):
    print('\033[96m', *message, '\033[0m')

class PrettyTabular(object):
    def __init__(self, head):
        self.head = head

    def head_string(self):
        line = ''
        for key, value in self.head.items():
            if 's' in value:
                dummy = value.format('0')
            else:
                dummy = value.format(0)
            span = max(len(dummy), len(key)) + 2
            key_format = '{:^' + str(span) + '}'
            line += key_format.format(key)
        return line

    def row_string(self, row_data):
        line = ''
        for key, value in self.head.items():
            data = value.format(row_data[key])
            span = max(len(key), len(data)) + 2
            line += ' ' * (span - len(data) - 1) + data + ' '
        return line

import shutil
import os
def create_folder(folder_name, exist_ok=False):
    if not exist_ok and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=exist_ok)