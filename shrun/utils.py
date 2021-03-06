# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,shrun//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Import

# %%
import os
import sys
import json
import functools
import subprocess
from pathlib import Path
from collections import OrderedDict

import dmenu


# %% [markdown]
#
# # Code

# %%
def show(iters, defs=None, prompt='Choose: '):
    try: 
        r = dmenu.show(iters, prompt=prompt)
        if not r : sys.exit()
        return r 
    except: raise Exception

def load_hist(path):
    data = []
    try:
        with open(str(path), 'r') as f:
            data = json.load(f)
    except FileNotFoundError: pass
    return data

def update_hist(path, hist, item):
    st = json.dumps(item)#, indent=4)
    hist_strs = [json.dumps(i) for i in hist]
    if st not in hist_strs:
        hist.append(item)
        with open(str(path), 'w') as f:
            json.dump(hist, f, indent=4)    

def get_field(hist, key): 
    seq = [i[key] for i in hist]
    return list(OrderedDict.fromkeys(seq))     

def port_map():
    HISTORY_FN = Path(os.environ['HOME']) / Path('.config/portmapper/hist.json')
    if not HISTORY_FN.parent.exists(): os.makedirs(str(HISTORY_FN.parent.absolute()))
    
    hist = load_hist(HISTORY_FN)
    
    def_users = get_field(hist, 'user')
    def_urls = get_field(hist, 'remote_host')
    def_local_ports = get_field(hist, 'local_port')
    def_host_ports = get_field(hist, 'host_port')
    
    user = show(def_users, prompt='Select USER: ')
    remote_host = show(def_urls, prompt='Select HOST: ')
    
    host_port = show(def_host_ports, prompt='Select HOST port: ')
    local_port = show(def_local_ports, prompt='Select LOCAL port: ')
    item =  {'remote_host':remote_host, 'user':user, 'host_port':host_port, 'local_port':local_port}
    
    update_hist(HISTORY_FN, hist, item)
    
    process = subprocess.Popen(['ssh', '-N', '-L', f'{local_port}:127.0.0.1:{host_port}', f'{user}@{remote_host}', '&'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
def multi_buf():
    colors = [
        '#282a2e',
        '#de935f',
        '#de935f',
        '#282a2e', 
    ]
    font_size = 14
    
    cmd = f'clipmenu -m 0 -fn monospace:size={font_size} -nb "{colors[0]}" -nf "{colors[1]}" -sb "{colors[2]}"  -sf "{colors[3]}"'
    #paste_ext = '&& xdotool click --clearmodifiers 2'
    subprocess.run(cmd, shell=True)
   


# %%
if __name__ == '__main__':
    #multi_buf()
    port_map()
    pass


# %%

# %%

# %%

# %%

# %%

# %%
