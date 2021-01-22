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
import os
import shutil
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from contextlib import contextmanager

import gdal
import numpy as np

# %% [markdown]
# # Todo
#
# - parallel r-w, not r
# - separate cutter from tiler
# - cutter only extracts patches (from image or image-folder)
# - tiler with zoom level
# - standalone saver
# - image-folder as src
# - webserver / zmq
#
#

# %%
@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def mp_func(foo, args, n):
    args_chunks = [args[i:i + n] for i in range(0, len(args), n)]
    with poolcontext(processes=n) as pool:
        res = pool.map(foo, args_chunks)
    return [ri for r in res for ri in r]

# %%
p = '/home/sokolov/work/tmp/aerial/data/fortBragg/fortBragg1018_1-1.tif'
#get_stats_gdal(p)

# %% jupyter={"outputs_hidden": true}
image_preprocessing_gdal(p, 'fortbragg', 8, block_size=(2048,2048))

# %%
# img/
# img_0_0.ext
# img_0_1.ext
# img_x_y.ext

# %%
_get_sub_grids(10,10,10,10)


# %%

# %%

# %%
def find_zoom(w,h, base_zoom=8):
    return max(0, int(np.ceil(np.log2(max(w,h))-base_zoom)))

def dump_info_json(name, w,h):
    d =  {
                'width':int(w),
                'height':int(h),
                'max_zoom':find_zoom(w,h),
                'tile_size':2048
            }
    with open(str(name), 'w') as f:
        json.dump(d, f)
        
def _write_block(block, name):
    x, y, block_data = block
    print(name, x,y,block_data.shape, block_data.dtype)
    t = Image.fromarray(block_data.transpose((1,2,0)))
    t.save(f'output/{name}_{x}_{y}.png')
    #raise NotImplementedError
    # DO SAVING


# %%
def select_block_position(name, block_size):
    nXBlocks, nYBlocks = _count_blocks(name, block_size=block_size)
    input_file, (W,H,*_), *_ = get_basics_gdal(name)
    
    for X in range(nXBlocks):
        if X == nXBlocks - 1: nXValid = W - X * block_size[0]
        myX = X * block_size[0]
        nYValid = block_size[1]
        for Y in range(nYBlocks):
            if Y == nYBlocks - 1: nYValid = H - Y * block_size[1]
            myY = Y * block_size[1]
            
            #block = reader_func(myX, myY, nXValid, nYValid)
            #input_file.ReadAsArray(xoff=myX, yoff=myY, xsize=nXValid, ysize=nYValid)
            yield myX, myY, nXValid, nYValid#X, Y, block
            
def image_preprocessing_gdal(input_name, 
                             output_name,
                             num_processes=1,
                             block_size=None,
                             image_processing_func=None,
                             percentile_cut=None,
                             show_tqdm=True):
    """ Takes image, process and saves it tile-wise with GDAL.
    
    Args:
        input_name (str): input image file name
        output_name (str): output file name to be created
        num_processes (int): number of parallel processes for stats calculations and image processsing.
                             None for maximum allowed by system. Defaults to 1

        image_processing_func (function, optional): should take numpy array (float32, [0,1]), tuple of image stats:
                                                    global_stats = (global_min, global_max, global_std, global_mean)
                                                    def image_processing_func(block, stats = global_stats):
                                                        pass
                                                    If number of channels after processing is less then input number,
                                                    one shoud use output_channels_override argument to specify it. 
                                                    Defaults to None.

        percentile_cut (int, optional): Percentile value to cut in np.clip. Used in stats.  Defaults to 1
        show_tqdm (bool, optional): Show tqdm progress bar

        Exmp:
                image_preprocessing_gdal(   'Fort1.tif',
                                            'Fort1_prep.tif',
                                            num_processes=None,
                                            image_processing_func=prep_func,
                                            percentile_cut=1)
    """
    if block_size is None:
        block_size = (256, 256)
    #global_stats = get_stats_gdal(input_name, perc_cut=percentile_cut, num_processes=num_processes)
    nXBlocks, nYBlocks = _count_blocks(input_name, block_size=block_size)

    # define subgrid size : 20x20 blocks, experimental
    sub_x_blocks, sub_y_blocks = 10, 10
    sub_grids = _get_sub_grids(sub_x_blocks, sub_y_blocks, nXBlocks, nYBlocks)
    r_args = [(input_name, sub_grid, block_size) for sub_grid in sub_grids]
    
    launch_mpq(_process_grid_blocks, r_args, _writer, (output_name, nXBlocks*nYBlocks), num_processes, show_tqdm)

    _, dims, *_  = get_basics_gdal(name)

def get_basics_gdal(name):
    """Get basic info from image
    Args:
        name (str): Image filename
    Returns:
        tuple: (file,                   File object (<gdal dataset>)
                dims,                   Dimensions, tuple: (1920,1024)
                bands_count,            Number of channels, int
                gdal_type,              Type via GDAL typage (gdal.GDT_Byte, etc)
                dtype  ,                Type via numpy dtype
                block_max_type_size     Max value of numpy dtype 
                )
    """
    file = gdal.Open(name, gdal.GA_ReadOnly)
    bands_count = file.RasterCount
    g_type = gdal.GetDataTypeName(file.GetRasterBand(1).DataType)
    dims = [file.RasterXSize, file.RasterYSize]
    _pix = file.ReadAsArray(0, 0, 1, 1)
    # dtype in numpy
    #block_max_type_size, type_np = _get_block_type_max_size(_pix)
    return file, dims, bands_count, g_type#, type_np, block_max_type_size

def _count_blocks(name, block_size=(256, 256)):
    # find total x and y blocks to be read
    _, dims, *_  = get_basics_gdal(name)
    nXBlocks = (int)((dims[0] + block_size[0] - 1) / block_size[0])
    nYBlocks = (int)((dims[1] + block_size[1] - 1) / block_size[1])
    return nXBlocks, nYBlocks

def _reader(arg, func, queue):
    gen = func(*arg)
    for i in gen:
        queue.put(i)

def _writer(output_name, total_blocks, queue):
    count = 0
    while(count < total_blocks):
        try:
            _write_block(queue.get(), output_name)
            count+=1
            queue.task_done()
        except mp.TimeoutError:
            print("timeout, quit.")
            break
        except Exception as e:
            print(e)
            break

def mp_func_wrapper(func, args):
    return func(*args)
    
def get_mp_func_args(func, args, ext):
    func_mp_args = [(arg, *ext) for arg in args]
    func_mp = partial(mp_func_wrapper, func)
    return func_mp, func_mp_args

def launch_mpq(r_func, r_args, w_func, w_args, num_processes, show_tqdm):
    m = mp.Manager()
    q = m.Queue(maxsize=50)
    if show_tqdm: pbar = tqdm(total=len(r_args))

    reader_mp, reader_mp_args = get_mp_func_args(_reader, r_args, (r_func, q))
    writer_args = *w_args, q

    with mp.Pool(num_processes) as p:    
        g = p.imap_unordered(reader_mp, reader_mp_args)
        writer_p = mp.Process(target=w_func, args=writer_args)
        writer_p.start()        
        for _ in g:
            if show_tqdm:
                pbar.update()
        
        writer_p.join()


# %%
def _process_grid_blocks(input_name, sub_grid, block_size, global_stats=None, image_processing_func=None):
    """  
    Read, process and write blocks, heart and soul of stream_reader
    """
    for x, y, block in _get_block_subgrid(input_name, sub_grid, block_size):
        #block = image_processing_func(block, stats=global_stats)
        yield x, y, block
        
def _get_block_subgrid(input_name, sub_grid, block_size):
    # block generator from subgrid
    x_start, x_n, y_start, y_n, nXBlocks, nYBlocks = sub_grid
    input_file, input_dims, *_ = get_basics_gdal(input_name)
    nXValid, nYValid = block_size[0], block_size[1]
    
    for X in range(x_start, x_start + x_n):
        if X == nXBlocks - 1:
            nXValid = input_dims[0] - X * block_size[0]
        # find X offset
        myX = X * block_size[0]
        # reset buffer size for start of Y loop
        nYValid = block_size[1]
        for Y in range(y_start, y_start + y_n):
            # change the block size of the final piece
            if Y == nYBlocks - 1:
                nYValid = input_dims[1] - Y * block_size[1]
            # find Y offset
            myY = Y * block_size[1]
            # reading data from band
            block = input_file.ReadAsArray(xoff=myX, yoff=myY, xsize=nXValid, ysize=nYValid)
            if block.ndim < 3:
                # one channel image
                block = np.expand_dims(block, 0)
            #yield myX, myY, block
            yield X, Y, block
            
            
def _get_sub_grids(nx_sub, ny_sub, nXBlocks, nYBlocks):
    """ Creates list of subgrid coords and sizes from size of one subgrid and whole grid
    
    Args:
        nx_sub (int): number of blocks, x
        ny_sub (int): number of blocks, y
        nXBlocks (int): total blocks, x
        nYBlocks (int): total blocks, y
    
    Returns:
        list: [[start_index_x, number_of_blocks_x, start_index_y, number_of_blocks_y, nXBlocks, nYBlocks],
               ...
               ]
    """
    sub_grids = []
    xr = nXBlocks // nx_sub + 1 if nx_sub < nXBlocks else 1
    yr = nYBlocks // ny_sub + 1 if ny_sub < nYBlocks else 1
    for x in range(xr):
        x_start = nx_sub * x
        x_n = nx_sub if x != nXBlocks // nx_sub else nXBlocks % nx_sub
        for y in range(yr):
            y_start = ny_sub * y
            y_n = ny_sub if y != nYBlocks // ny_sub else nYBlocks % ny_sub
            sub_grids.append([x_start, x_n, y_start, y_n, nXBlocks, nYBlocks])
    return sub_grids

def get_stats_gdal(name, perc_cut=1, block_size=None, num_processes=1):
    """[summary]
        name (str): Image filename
        perc_cut (int, optional): Number of percents to cut image values from. Defaults to 1.
        block_size ((int, int)), optional): Size of reading window. Defaults to None.
        num_processes (int, optional): Number of processes for MP via butil.common.launch_multiprocess. Defaults to 1.
    
    Returns:
        tuple: Image stats, but not 100% presise (calculated on subsamples of data)
               (global_min,     
                global_max,
                global_std,
                global_mean) 

    """
    if block_size is None:
        _, dims, *_ = get_basics_gdal(name)
        block_size = (256, 256) if dims[0]*dims[1] > 16e6 else (32, 32)
        
    nXBlocks, nYBlocks = _count_blocks(name, block_size=block_size)
    if num_processes == 1:
        subgrid = 0, nXBlocks, 0, nYBlocks, nXBlocks, nYBlocks
        stats = _get_stats_subgrid(name, subgrid, block_size)
    else:
        # subgrid size of 20x20 blocks, experimental
        subgrids = _get_sub_grids(20, 20, nXBlocks, nYBlocks)
        args = [(name, subgrid, block_size) for subgrid in subgrids]
        #foo = partial(mp_func_wrapper, _get_stats_subgrid)
        stats = mp_func(_get_stats_subgrid, args, num_processes)#launch_multiprocess(foo, args, num_cores=num_processes)
        stats = np.array([stat_line for stat in stats for stat_line in stat])

    mins, maxs = stats[:, 0], stats[:, 1]
    stats_cliped = stats[(mins > np.percentile(mins, perc_cut)) & (maxs < np.percentile(maxs, 100-perc_cut))]
    global_min = np.percentile(mins, perc_cut)
    global_max = np.percentile(maxs, 100-perc_cut)
    global_std = stats_cliped[:, 2].std()
    global_mean = stats_cliped[:, 2].mean()
    return global_min, global_max, global_std, global_mean

def _get_stats_subgrid(name, sub_grid, block_size):
    nx_blocks, ny_blocks = sub_grid[1], sub_grid[3]
    total_blocks = nx_blocks * ny_blocks
    stats = np.zeros((total_blocks, 3))
    gen = _get_block_subgrid(name, sub_grid, block_size=block_size)
    for i in range(total_blocks):
        _, _, block = next(gen)
        x_rand, y_rand = np.random.randint(0, block.shape[1]), np.random.randint(0, block.shape[2])
        min_ = block[:3,x_rand, y_rand].min()
        max_ = block[:3,x_rand, y_rand].max()
        pix = block[0, x_rand, y_rand]
        stats[i] = np.array([min_, max_, pix])
    return stats


def clip_block(block , min_, max_):
    block = block.astype(np.float32)
    block = (block - min_) / (max_ - min_)
    return np.clip(block, 0, 1)

# %%

# %%

# %%
