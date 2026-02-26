import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import cv2
import wave, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import pandas as pd
import matplotlib.gridspec as gridspec
import ray
from ray.actor import ActorHandle
from tqdm.auto import tqdm
import matplotlib as mpl
import tifffile
mpl.rcParams.update({'font.size':10})

import fm2p

plasma_map = plt.cm.plasma(np.linspace(0,1,15))
cat_cmap = {
    'movement': plasma_map[12,:],
    'early': plasma_map[10,:],
    'late': plasma_map[8,:],
    'biphasic': plasma_map[5,:],
    'negative': plasma_map[2,:],
    'unresponsive': 'dimgrey'
}

def avi_to_arr(path, ds=0.25, max_frames=None):
    vid = cv2.VideoCapture(path)
    # array to put video frames into
    # will have the shape: [frames, height, width] and be returned with dtype=int8
    arr = np.empty([int(vid.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)*ds),
                        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)*ds)], dtype=np.uint8)
    # iterate through each frame
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    for f in range(0, n_frames):
        # read the frame in and make sure it is read in correctly
        ret, img = vid.read()
        if not ret:
            break
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # downsample the frame by an amount specified in the config file
        img_s = cv2.resize(img, (0,0), fx=ds, fy=ds, interpolation=cv2.INTER_NEAREST)
        # add the downsampled frame to all_frames as int8
        arr[f,:,:] = img_s.astype(np.int8)
    return arr

def drop_repeat_sacc(eventT, onset=True, win=0.020):
    """For saccades spanning multiple camera
    frames, only keep one saccade time. Either first or last.

    If `onset`, keep the first in a sequence (i.e. the onset of
    the movement). otherwise, keep the final saccade in the sequence

    """
    duplicates = set([])
    for t in eventT:
        if onset:
            new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
        else:
            new = eventT[((t-eventT)<win) & ((t-eventT)>0)]
        duplicates.update(list(new))
    out = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    return out

from asyncio import Event
from typing import Tuple
from time import sleep

@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter

class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return
            
import h5py
import datetime
import numpy as np
import pandas as pd
def write_h5(filename, dic):
    """ Write a dictionary to an .h5 file.

    The dictionary can only contain values that are of the
    following types: dict, list, numpy.ndarray, or basic scalar
    types (int, float, str, bytes). The hierarchy of the dictionary
    is preserved in the .h5 file that is written. The keys of
    the dictionary can only be type str (not int).

    Parameters
    ----------
    filename : str
        Path to the .h5 file.
    dic : dict
        Dictionary to be saved.

    Notes
    -----
    Modified from https://codereview.stackexchange.com/a/121308

    """

    with h5py.File(filename, 'w') as h5file:

        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):

    if isinstance(dic,dict):
        iterator = dic.items()

    elif isinstance(dic,list):
        iterator = enumerate(dic)

    else:
        ValueError('Cannot save %s type' % type(dic))

    for key, item in iterator:

        if isinstance(dic,list):
            key = str(key)
            
        if isinstance(item, (np.ndarray, np.int16, np.int64, np.float64, int, float, str, bytes, np.float32, np.int32)):
            
            try:
                h5file[path + key] = item
            
            except TypeError:
                if isinstance(item, np.ndarray) and (item.dtype == object):
                    recursively_save_dict_contents_to_group(h5file, path + key + '/', item.item())

        elif isinstance(item, dict) or isinstance(item, list):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)

        elif isinstance(item, datetime.datetime):
             h5file[path + key] = fm2p.time2str(item)

        else:
            raise ValueError('Cannot save %s type'%type(item))


def read_h5(filename, ASLIST=False):
    """ Read an .h5 file in as a dictionary.

    Parameters
    ----------
    filename : str
        Path to the .h5 file.
    ASLIST : bool
        If True, the dictionary will be read in as a list (on the first
        layer). Keys must have been convertable to integers when the file
        was written.

    Notes
    -----
    Modified from https://codereview.stackexchange.com/a/121308

    """
    with h5py.File(filename, 'r') as h5file:
        out = recursively_load_dict_contents_from_group(h5file, '/')
        if ASLIST:
            outl = [None for l in range(len(out.keys()))]
            for key, item in out.items():
                outl[int(key)] = item
            out = outl
        return out


def recursively_load_dict_contents_from_group(h5file, path):
    
    ans = {}

    for key, item in h5file[path].items():

        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]

        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file,
                                                                 path + key + '/')

    return ans



def get_group_h5_keys(savepath):
    """ Get the keys of a group h5 file.

    This will list the keys (i.e. the session names) of an h5 file
    written by the function write_group_h5 (above). It does not need
    to read the entire file into memory to check these values.

    Parameters
    ----------
    savepath : str
        Path to the .h5 file.
    
    Returns
    -------
    keys : list
        List of keys (i.e. session names) in the h5 file.

    """

    with pd.HDFStore(savepath) as hdf:
        
        keys = [k.replace('/','') for k in hdf.keys()]

    return keys


def read_group_h5(path, keys=None):
    """ Read a group h5 file.

    This will read in a group h5 file written by the function
    write_group_h5 (above). It will read in all keys and stack
    them into a single dataframe. Alternatively, you can specify
    a list of keys to read in from the keys present, and only those
    recordings will be read into memory and stacked together.
    
    Parameters
    ----------
    path : str
        Path to the .h5 file.
    keys : list or str (optional).
        List of keys (i.e. session names) in the h5 file. If None,
        all keys will be read in.
    
    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing all data from the h5 file.

    """

    if type(keys) == str:

        df = pd.read_hdf(path, keys)

        return df
    
    if keys is None:

        keys = get_group_h5_keys(path)

    dfs = []
    for k in sorted(keys):

        _df = pd.read_hdf(path, k) 
        dfs.append(_df)

    df = pd.concat(dfs)

    return df


base_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251024_DMM_DMM061_pos08/fm1'

print('Reading preproc data')
preproc_path = os.path.join(base_dir, '251024_DMM_DMM061_fm_01_preproc.h5')
data = read_h5(preproc_path)

twopT = data['twopT']
eyeT_trim = data['eyeT_trim']
eyeT_startInd = int(data['eyeT_startInd'])

print('Reading videos')
# Left video: fm01_0001.mp4
top_vid_path = os.path.join(base_dir, 'fm01_0001.mp4')
top_vid = avi_to_arr(top_vid_path, ds=1) # Keep resolution or ds if needed

# Middle video: eyecam
eye_vid_path = os.path.join(base_dir, '251024_DMM_DMM061_fm_01_eyecam_deinter.avi')
eye_vid = avi_to_arr(eye_vid_path, ds=1)

# Right video: tif stack
tif_path = os.path.join(base_dir, 'file_00001.tif')
print('Reading TIF stack')
tif_stack = tifffile.imread(tif_path)

# Apply rolling average of 6 to tif stack
print('Applying rolling average to TIF')
def rolling_average(arr, window=6):
    # Simple moving average along axis 0
    # We'll use a convolution or similar. Since it's a stack (T, H, W), we iterate or use scipy
    # Using a simple loop for memory efficiency or numpy tricks
    ret = np.cumsum(arr.astype(float), axis=0)
    ret[window:] = ret[window:] - ret[:-window]
    return (ret[window - 1:] / window).astype(arr.dtype)

# Pad to keep length if necessary, or just trim timestamps. 
# Usually rolling average shrinks array. Let's pad with edge values or just trim.
# The prompt says "with a rolling average of 6 applied. that also uses the twopT timetamps".
# Assuming 1-to-1 mapping, we should try to preserve shape or adjust twopT.
# Simple boxcar filter 'same' mode equivalent:
from scipy.ndimage import uniform_filter1d
tif_stack_smooth = uniform_filter1d(tif_stack.astype(float), size=6, axis=0).astype(np.uint8)
# Or strictly causal? "rolling average" usually implies causal or centered. 
# Let's stick to uniform_filter1d which is centered by default, good for alignment.


@ray.remote
def plot_frame_img(currentT, top_vid, eye_vid, tif_stack, twopT, eyeT_trim, eyeT_startInd, pbar:ActorHandle):

    # Find indices
    # Top and Tif use twopT
    twopFr = np.argmin(np.abs(twopT - currentT))
    
    # Eye uses eyeT_trim, then offset by eyeT_startInd
    eyeTrimFr = np.argmin(np.abs(eyeT_trim - currentT))
    eyeFr = eyeT_startInd + eyeTrimFr

    # Get images
    # Handle bounds
    if twopFr >= len(top_vid): twopFr = len(top_vid) - 1
    if twopFr >= len(tif_stack): twopFr = len(tif_stack) - 1
    if eyeFr >= len(eye_vid): eyeFr = len(eye_vid) - 1

    img_top = top_vid[twopFr]
    img_eye = eye_vid[eyeFr]
    img_tif = tif_stack[twopFr]

    # Normalize TIF for display if needed (it's likely uint16 or similar, convert to uint8)
    if img_tif.dtype != np.uint8:
        # Simple min-max norm for display
        img_tif = img_tif.astype(float)
        img_tif = ((img_tif - np.min(img_tif)) / (np.max(img_tif) - np.min(img_tif) + 1e-5) * 255).astype(np.uint8)

    # Calculate dimensions for tight side-by-side stacking
    h1, w1 = img_top.shape
    h2, w2 = img_eye.shape
    h3, w3 = img_tif.shape
    
    ar1 = h1 / w1
    ar2 = h2 / w2
    ar3 = h3 / w3
    
    # Create figure with no spacing, keeping aspect ratio.
    # We set a fixed height and calculate the width needed to place
    # the videos side-by-side without distorting them.
    fig_h = 5 # inches
    fig_w = fig_h * (1/ar1 + 1/ar2 + 1/ar3)
    
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1/ar1, 1/ar2, 1/ar3])
    gs.update(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax1.imshow(img_top, cmap='gray')
    ax1.axis('off')
    
    ax2.imshow(img_eye, cmap='gray')
    ax2.axis('off')

    ax3.imshow(img_tif, cmap='gray')
    ax3.axis('off')

    width, height = fig.get_size_inches() * fig.get_dpi()
    fig.canvas.draw() # draw the canvas, cache the renderer
    images = np.frombuffer(fig.canvas.tostring_rgb(),
                    dtype='uint8').reshape(int(height), int(width), 3)
    
    plt.close()
    pbar.update.remote(1)
    return images


t_start = 0
tlen = 3 * 60 # 3 minutes

# Output video settings
out_fps = 60
playback_speed = 2.0

# Total output frames
num_out_frames = int(tlen * out_fps / playback_speed)

print(f"Generating {num_out_frames} frames for {tlen}s of data at {playback_speed}x speed.")

print('Creating ray actors')
mpl.use('agg')

pb = ProgressBar(num_out_frames)
actor = pb.actor

top_vid_r = ray.put(top_vid)
eye_vid_r = ray.put(eye_vid)
tif_stack_r = ray.put(tif_stack_smooth)
twopT_r = ray.put(twopT)
eyeT_trim_r = ray.put(eyeT_trim)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Loop over parameters appending process ids
print('Looping over parameters and appending proces ids')
result_ids = []

for f in range(num_out_frames):
    # Calculate current time in data
    # Frame f corresponds to time t = f * (playback_speed / out_fps)
    currentT = t_start + f * (playback_speed / out_fps)
    
    result_ids.append(plot_frame_img.remote(
        currentT, top_vid_r, eye_vid_r, tif_stack_r, twopT_r, eyeT_trim_r, eyeT_startInd, actor
    ))
    
savepath = 'time_aligned_demo_3min_2x.mp4'

out = None
    

print('Writing video frames')
# Progressbar and write results incrementally to save memory
for i, res_id in enumerate(tqdm(result_ids, desc="Rendering and writing video")):
    frame = ray.get(res_id)
    result_ids[i] = None # Release reference to free Ray object store memory
    
    if out is None:
        height, width, _ = frame.shape
        out = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (width, height))
    
    # Frame is RGB (from matplotlib), OpenCV expects BGR
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    del frame

if out is not None:
    out.release()

print('Video file released')
