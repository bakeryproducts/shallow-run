
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/utils.ipynb

import os
import subprocess
from pathlib import Path

def multi_buf():
    colors = [
        '#282a2e',
        '#de935f',
        '#de935f',
        '#282a2e',
    ]
    font_size = 14

    cmd = f'clipmenu -m 0 -fn monospace:size={font_size} -nb "{colors[0]}" -nf "{colors[1]}" -sb "{colors[2]}"  -sf "{colors[3]}"'
    paste_ext = '&& xdotool click --clearmodifiers 2'
    subprocess.run(cmd, shell=True)



if __name__ == '__main__':
    multi_buf()