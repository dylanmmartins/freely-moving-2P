
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

POST_PATH = ('/home/dylan/goard-nas1/Raw_Data_Archives/Mini2P_data/'
             '260406_DMM_DMM070_pos03/file_00002.tif')
PRE_PATH  = '/home/dylan/goard-nas1/Raw_Data_Archives/Mini2P_data/260129_DMM_DMM063_pos17/file_00002.tif'

FRAME_RATE   = 7.5
N_MINUTES    = 5
N_FRAMES     = int(N_MINUTES * 60 * FRAME_RATE)
OUT_PATH     = 'frame_mean_comparison.svg'


def load_frame_means(tif_path: str, n_frames: int) -> np.ndarray:
    """Return per-frame mean pixel value for the first n_frames of a tif."""
    tif   = tifffile.TiffFile(tif_path)
    total = min(n_frames, len(tif.pages))
    means = np.empty(total, dtype=np.float64)
    for i in range(total):
        means[i] = tif.pages[i].asarray().sum()
    tif.close()
    return means


def baseline_pct(trace: np.ndarray) -> np.ndarray:
    """Frame sum as % of the t=0 baseline (100 = no change)."""
    return trace / trace[0] * 100.0


print('Loading post-power-box ...')
post_means = load_frame_means(POST_PATH, N_FRAMES)
print('Loading pre-power-box ...')
pre_means  = load_frame_means(PRE_PATH,  N_FRAMES)

post_pct = baseline_pct(post_means)
pre_pct  = baseline_pct(pre_means)

t_post = np.arange(len(post_pct)) / FRAME_RATE / 60.0
t_pre  = np.arange(len(pre_pct))  / FRAME_RATE / 60.0

plt.rcParams.update({
    'font.size':        13,
    'axes.titlesize':   14,
    'axes.labelsize':   13,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
})

fig, ax = plt.subplots(figsize=(5, 3.5))

ax.plot(t_post, post_pct, color='tab:blue',   lw=1.0, alpha=0.85,
        label='post-power-box')
ax.plot(t_pre,  pre_pct,  color='tab:orange', lw=1.0, alpha=0.85,
        label='pre-power-box')

ax.axhline(100, color='0.6', lw=0.7, ls='--')
# ax.set_yscale('log')
ax.set_xlabel('time (min)')
ax.set_ylabel('frame sum\n(% of t=0 baseline)')
ax.set_xlim(0, N_MINUTES)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
fig.savefig(OUT_PATH.replace('.svg', '.png'), dpi=200, bbox_inches='tight')
print(f'Saved: {OUT_PATH}')
