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

# %%
# #!/usr/bin/env python3
import sys
from pathlib import Path
import subprocess


def main():
    save_path = 'src' if len(sys.argv) < 2 else sys.argv[1]
    cfg = Path('./.shconv')
    with open(cfg, 'r') as f:
        files = f.read()
    files = [f for f in files.split('\n') if f]

    #cmds = [f'echo {f} | entr shn2s {f} {save_path}' for f in files]
    cmds = [f'echo {f} | entr python3 -m shrun.n2s {f} {save_path}' for f in files]
    
    print(f'Watching for {files}')
    p = subprocess.Popen(' & '.join(cmds), shell=True)
    p.communicate()
    print(files)

if __name__ == '__main__':
    main()

# %%
# entr : https://github.com/eradman/entr/

# %%

# %%

# %%

# %%

# %%
