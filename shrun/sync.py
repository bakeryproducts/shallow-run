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

# %% tags=["active-ipynb"]
# %load_ext autoreload
# %autoreload 2

# %%
import os
import sys
import json
import subprocess
import importlib
from pathlib import Path
from functools import partial


# %% [markdown]
# # Code

# %%
class SyncerBase:
    def __init__(self, sender, reciever, kwargs): self.sender, self.reciever, self.kwargs = sender, reciever, kwargs
    def push(self, src, dst): self.sender(**self.kwargs, src=src, dst=dst)
    def pull(self, src, dst): self.reciever(**self.kwargs, src=src, dst=dst)
        
def init_custom_syncer(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("Syncer", path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo.Syncer


# %%
def main():
    """
    Sample local config:
        {
            1:{ 
            
                'syncer':'extra/syncer.py'# not neccesary
                'local':'output/test.tar',
                'remote':'ftp.host.com:21:files/sync/Project1/test.tar'
               
               },
        }
        1 - file_id, alias for file
        
    usage: 
        1. Without sycner class, using existing sftp:
            shsync push 1 shrun.sftp
        2. With sync class from cfg file: shsync pull 1 
    """
    
    assert len(sys.argv) >=2
    action = sys.argv[1]
    assert action in ['push', 'pull']
    file_id = sys.argv[2]
    
    cfg_path = Path('./.shsync')
    with open(str(cfg_path), 'r') as f:
        cfg = json.load(f)
    assert file_id in cfg, (file_id, cfg)
    
    if len(sys.argv) == 4:
        syncer = importlib.import_module(sys.argv[3]).Syncer
    else:
        syncer_path = cfg[file_id]['syncer']
        syncer = init_custom_syncer(syncer_path)
    
    local, remote = cfg[file_id]['local'], cfg[file_id]['remote']
    local = str(Path().absolute() / local)
    
    if action == 'push':
        print(f'\n\nPushing {local} to {remote}\n\n')
        syncer.push(src=local, dst=remote)
    else:
        print(f'\n\nPulling {remote} to {local}\n\n')
        syncer.pull(src=remote, dst=local)
    
    
if __name__ == '__main__':
    main()

# %%

# %% [markdown]
# # Tests

# %% tags=["active-ipynb"]
# d = {
#          1:{ 
#             'syncer':'extra/syncer.py',
#             'local':'output/test.tar',
#             'remote':'ftp.host.com:21:files/sync/Project1/test.tar'
#            },
# }
# with open("tests/.shsync", 'w') as f:
#     json.dump(d, f, indent=4)

# %% tags=["active-ipynb"]
# !cd tests && nb_run.py pull 1

# %% tags=["active-ipynb"]
# !cd tests && nb_run.py push 1

# %%

# %%
