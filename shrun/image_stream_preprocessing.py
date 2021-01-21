import os
import shutil
from functools import partial
import numpy as np
import gdal
import  multiprocessing as mp
from tqdm import tqdm

from butil.common import mp_func_wrapper, launch_multiprocess
from butil.common import create_or_delete_and_create_folder, load_csv, get_total_megapixels
from sampler.file_ds import load_dataset


def get_stats_gdal(name, perc_cut=1, block_size=None, num_processes=1):
    """[summary]
    
    Args:
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
    #st = time.time()
    if block_size is None:
        _, dims, *_ = get_basics_gdal(name)
        # block size on small images should be smallish
        block_size = (256, 256) if dims[0]*dims[1] > 16e6 else (32, 32)
        
    nXBlocks, nYBlocks = _count_blocks(name, block_size=block_size)
    #print(block_size, nXBlocks, nYBlocks)
    if num_processes == 1:
        subgrid = 0, nXBlocks, 0, nYBlocks, nXBlocks, nYBlocks
        stats = _get_stats_subgrid(name, subgrid, block_size)
    else:
        # subgrid size of 20x20 blocks, experimental
        subgrids = _get_sub_grids(20, 20, nXBlocks, nYBlocks)
        args = [(name, subgrid, block_size) for subgrid in subgrids]
        foo = partial(mp_func_wrapper, _get_stats_subgrid)
        stats = launch_multiprocess(foo, args, num_cores=num_processes)
        stats = np.array([stat_line for stat in stats for stat_line in stat])

    mins, maxs = stats[:, 0], stats[:, 1]
    stats_cliped = stats[(mins > np.percentile(mins, perc_cut)) & (maxs < np.percentile(maxs, 100-perc_cut))]
    global_min = np.percentile(mins, perc_cut)
    global_max = np.percentile(maxs, 100-perc_cut)
    global_std = stats_cliped[:, 2].std()
    global_mean = stats_cliped[:, 2].mean()
    #print(time.time() - st)
    return global_min, global_max, global_std, global_mean

def _get_stats_subgrid(name, sub_grid, block_size):
    # get stats from part of image
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

def _get_block_type_max_size(block):
    """ Get max value of size for block type 
    
    Args:
        block (np.array): input data
    
    Returns:
        tuple, max_size, type
    
    Ex.: 
            max_size, type = _get_block_type_max_size(np.zeros(3, dtype=np.uint8))
            maxsize == 255
            type = dtype('uint8')
    """
    
    try:
        # interger type
        max_value = np.iinfo(block.dtype).max
    except ValueError:
        # floating point type
        np.finfo(block.dtype).max
        max_value = 1
    except Exception as e:
        print(e,'Type conversion gone wrong')
    return max_value, block.dtype
        
def create_blank_tiff(input_name, output_name, output_channels_override, output_bit_override):
    _, input_dims, input_bands_count, input_type, *_ = get_basics_gdal(input_name)
    drv = gdal.GetDriverByName('GTiff')
    gdal_type = gdal.GDT_Byte if output_bit_override else gdal.GetDataTypeByName(input_type)
    bands_count = input_bands_count if not output_channels_override else output_channels_override
    file = drv.Create(output_name, input_dims[0], input_dims[1], bands_count, gdal_type)
    # to create file with rastersize on disk, write something in it:
    band = file.GetRasterBand(1)
    band.WriteArray(np.random.random((1, 1)) , xoff=0, yoff=0)  
    del file
      
def _process_grid_blocks(input_name, sub_grid, block_size,
                        global_stats, image_processing_func, output_bit_override):
    """  
    Read, process and write blocks, heart and soul of stream_reader

    """
    #output_file, _, out_bands_count, *_ = get_basics_gdal(output_name, update=True)
    #print('READING SUBGRID', sub_grid)
    for x, y, block in _get_block_subgrid(input_name, sub_grid, block_size):
        block = image_processing_func(block, stats=global_stats)
        if block.ndim < 3:
            raise ValueError('''\t Block cant have < 3 dims (got {0}) for processing into
                                1 channel use preprocessing function
                                with output shape of (1,...):'''.format(block.ndim))

        if output_bit_override:
            # block dtype may have been changed in preprocessing
            block_max_type_size, _ = _get_block_type_max_size(block)
            block = (255. * block / block_max_type_size).astype(np.uint8)            
        
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
            yield myX, myY, block
            
def image_preprocessing_gdal(input_name, 
                             output_name,
                             num_processes=1,
                             image_processing_func=None,
                             percentile_cut=1,
                             output_bit_override=False,
                             output_channels_override=None,
                             show_tqdm=True):
    """ Takes image, process and saves it tile-wise with GDAL.
        Output images will be same size, channel and pixel-type
        like original image (unless specified with override flags), but .tif ext.
        Using multiprocessing via butil.common.launch_multiprocess.
        Created for memory safe preprocessing of large images.
    
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
        output_bit_override (bool, optional):   If true, after all processing: if data block dtype:
                                                                            1) uint8: pass
                                                                            2) uint16: data = data*255/65536
                                                                            3) float: data *= 255 
                                                Then block converts to type np.uint8

        output_channels_override (int, optional): See image_processing_func for using this. Defaults to None.
        show_tqdm (bool, optional): Show tqdm progress bar

        Exmp:
                image_preprocessing_gdal(   'Fort1.tif',
                                            'Fort1_prep.tif',
                                            num_processes=None,
                                            image_processing_func=prep_func,
                                            percentile_cut=1,
                                            output_bit_override=False,
                                            output_channels_override=None)
      
    """
    
    if not image_processing_func:
        raise ValueError('\t Please specify preprocessing function or grayscaling')
    
    def check_exst_name(name):
        if os.path.isfile(name):
            os.remove(name)
    
    check_exst_name(output_name)
    output_name_base, output_name_ext = os.path.splitext(output_name)
    if output_name_ext != '.tif':
        output_name = output_name_base + '.tif'
        #print('\t\t Changing output name to {}'.format(output_name))
    check_exst_name(output_name)
    
    create_blank_tiff(input_name, output_name, output_channels_override, output_bit_override)
    
    # define blocksize for reading/writing images. Can be different, but sizes over 256 doesnt change much
    block_size = (256, 256)
    
    global_stats = get_stats_gdal(input_name, perc_cut=percentile_cut, num_processes=num_processes)
    nXBlocks, nYBlocks = _count_blocks(input_name, block_size=block_size)

    if num_processes == 1:
        sub_grid = 0, nXBlocks, 0, nYBlocks, nXBlocks, nYBlocks
        
        gen = _process_grid_blocks(input_name, sub_grid, block_size, 
                           global_stats, image_processing_func, output_bit_override)
        
        output_file, _, out_bands_count, *_ = get_basics_gdal(output_name, update=True)
        for block in gen:
            _write_block(block, output_file, out_bands_count)
        
    else:
        # define subgrid size : 20x20 blocks, experimental
        sub_x_blocks, sub_y_blocks = 10, 10
        sub_grids = _get_sub_grids(sub_x_blocks, sub_y_blocks, nXBlocks, nYBlocks)
        
        r_args = [(input_name, sub_grid, block_size, 
                global_stats, image_processing_func, output_bit_override) for sub_grid in sub_grids]
        
        launch_mpq(_process_grid_blocks, r_args, _writer, (output_name, nXBlocks*nYBlocks), num_processes, show_tqdm)

def apply_gray(patch, mode='BT601'):
        if patch.ndim < 3 or patch.shape[0] < 2:
            #raise ValueError('cant grayscale block with shape {}'.format(patch.shape))
            return patch
        
        # different algorithms for grayscaling
        if mode=='average':
            return np.expand_dims(patch.mean(axis=0),0)
        elif mode == 'BT601':
            _cs = np.array([0.114, 0.587, 0.299], dtype=np.float32)
            patch[0,...] *= _cs[0]
            patch[1,...] *= _cs[1]
            patch[2,...] *= _cs[2]
            return np.expand_dims(patch.sum(axis=0),0)
        else:
            raise Exception 

def clip_block(block , min_, max_):
    block = block.astype(np.float32)
    block = (block - min_) / (max_ - min_)
    return np.clip(block, 0, 1)

def get_basics_gdal(name, update=False):
    """Get basic info from image
    
    Args:
        name (str): Image filename
        update (bool, optional): Allows to make changes in image with 
                                 returned image_file descriptor object. Defaults to False.
    
    Returns:
        tuple: (file,                   File object (<gdal dataset>)
                dims,                   Dimensions, tuple: (1920,1024)
                bands_count,            Number of channels, int
                gdal_type,              Type via GDAL typage (gdal.GDT_Byte, etc)
                dtype  ,                Type via numpy dtype
                block_max_type_size     Max value of numpy dtype 
                )
    """
    if update:
        file = gdal.Open(name, gdal.GA_Update)
    else:
        file = gdal.Open(name, gdal.GA_ReadOnly)
    bands_count = file.RasterCount
    g_type = gdal.GetDataTypeName(file.GetRasterBand(1).DataType)
    dims = [file.RasterXSize, file.RasterYSize]
    _pix = file.ReadAsArray(0, 0, 1, 1)
    # dtype in numpy
    block_max_type_size, type_np = _get_block_type_max_size(_pix)
    return file, dims, bands_count, g_type, type_np, block_max_type_size

def _count_blocks(name, block_size=(256, 256)):
    # find total x and y blocks to be read
    _, dims, *_  = get_basics_gdal(name)
    nXBlocks = (int)((dims[0] + block_size[0] - 1) / block_size[0])
    nYBlocks = (int)((dims[1] + block_size[1] - 1) / block_size[1])
    return nXBlocks, nYBlocks

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

def _reader(arg, func, queue):
    gen = func(*arg)
    for i in gen:
        queue.put(i)

def _writer(output_name, total_blocks, queue):
    output_file, _, out_bands_count, *_ = get_basics_gdal(output_name, update=True)
    #print('writer')
    count = 0
    while(count < total_blocks):
        try:
            _write_block(queue.get(), output_file, out_bands_count)
            count+=1
            queue.task_done()
        except mp.TimeoutError as toe:
            print("timeout, quit.")
            break
        except Exception as e:
            print(e)
            break
    #print('Done')
    output_file = None

def _write_block(block, file, bands_count):
    x, y, block_data = block
    for i_band in range(bands_count):
        band = file.GetRasterBand(i_band + 1)
        band.WriteArray(block_data[i_band], xoff=x, yoff=y)
        band = None

def get_mp_func_args(func, args, ext):
    func_mp_args = [(arg, *ext) for arg in args]
    func_mp = partial(mp_func_wrapper, func)
    return func_mp, func_mp_args

def launch_mpq(r_func, r_args, w_func, w_args, num_processes, show_tqdm):
    m = mp.Manager()
    q = m.Queue(maxsize=50)

    if show_tqdm:
        pbar = tqdm(total=len(r_args))

    reader_mp, reader_mp_args = get_mp_func_args(_reader, r_args, (r_func, q))
    writer_args = *w_args, q

    with mp.Pool(num_processes) as p:    
        g = p.imap_unordered(reader_mp, reader_mp_args)
        writer_p = mp.Process(target=w_func, args=writer_args)
        writer_p.start()        
        for _ in g:
            if show_tqdm:
                pbar.update()
        
        #w_func(*writer_args)
 
        writer_p.join()

