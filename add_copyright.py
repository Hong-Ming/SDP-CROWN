"""Script for adding a copyright header to every file"""

import os


divider = '######################'


def get_header(path):
    with open(path) as file:
        lines = file.readlines()
    assert lines[0].startswith(divider)
    for i in range(1, len(lines)):
        if lines[i].startswith(divider):
            return ''.join(lines[:i+1])
    raise RuntimeError


def process(folder, header, reference):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isdir(path):
            if filename != 'auto_LiRPA':
                process(path, header, reference)
        elif path.endswith('.py'):
            if path == reference:
                continue
            else:
                with open(path) as file:
                    lines = file.readlines()
                if len(lines) == 0:
                    continue
                if lines[0].startswith(divider):
                    for i in range(1, len(lines)):
                        if lines[i].startswith(divider):
                            lines = lines[i+1:]
                            break
                print('Adding the copyright header to', path)
                with open(path, 'w') as file:
                    file.write(header)
                    file.write(''.join(lines))


reference = 'auto_LiRPA/bound_general.py'
header = get_header(reference)
print('Header for auto_LiRPA:')
print(header)
process('auto_LiRPA', header, reference)

reference = 'complete_verifier/abcrown.py'
header = get_header(reference)
print('Header for alpha-beta-CROWN:')
print(header)
process('complete_verifier', header, reference)
