
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/run.ipynb

import os
import sys
import json
from pathlib import Path
import subprocess
from functools import partial

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

def main():
    """
    Sample local config:
        {
            1:{

                'syncer':'extra/syncer.py'
                'local':'output/test.tar',
                'remote':'ftp.host.com:21:files/sync/Project1/test.tar'

               },
        }
        1 - alias for file

    usage: nb_run.py push 1
           nb_run.py pull 1
    """

    assert len(sys.argv) >=2
    action = sys.argv[1]
    assert action in ['push', 'pull']
    file_id = sys.argv[2]


    cfg_path = Path('./.shsync')
    with open(str(cfg_path), 'r') as f:
        cfg = json.load(f)
    assert file_id in cfg, (file_id, cfg)

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