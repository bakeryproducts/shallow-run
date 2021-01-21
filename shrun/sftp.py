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
from functools import partial

from shrun import sync


# %% [markdown]
# # Code

# %%
def init_sftp_args():
    sftp_args = {
        'password': os.environ['SFTPPASS'],
        'username': os.environ['SFTPUSER'],
    }
    return sftp_args

def sftp(password, username, src, dst, action):
    import paramiko
    if action == 'push': host, port, dst = dst.split(':', 2)
    else: host, port, src = src.split(':', 2)
    
    transport = paramiko.Transport((host, int(port)))
    transport.connect(username = username, password = password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    
    if action == 'pull': sftp.get(src, dst)
    elif action == 'push': sftp.put(src, dst)
    
    sftp.close()
    transport.close()

pusher = partial(sftp, action='push')
puller = partial(sftp, action='pull')

Syncer = sync.SyncerBase(sender=pusher, reciever=puller, kwargs=init_sftp_args())

# %%

# %%
