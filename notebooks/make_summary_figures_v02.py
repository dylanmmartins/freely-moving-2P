import os
import sys

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'manuscript_figures', 'figure_data.h5')
OUT = os.path.join(HERE, 'manuscript_figures_py_v02')
FORMATS = ('png', 'pdf')
DPI = 300

WIDTH = 8.5                    # layout width (in); figures are shrunk to OUT_WIDTH when saved
OUT_WIDTH, OUT_MAX_HEIGHT = 6.5, 8.0   # page size limits (in)
SCALE = OUT_WIDTH / WIDTH
MAX_HEIGHT = OUT_MAX_HEIGHT / SCALE    # tallest layout that fits (10.46 in)
BAR = 0.125                    # bar width in layout inches (0.096 in on the page)
FONT = 9.07
LABEL = 1.2 * FONT
LETTER = 1.5 * FONT
ERR_LW = 1.42
AREAS = ['V1', 'RL', 'A', 'AM']
AREA_COLOR = {'V1': '#1f77b4', 'RL': '#ff7f0e', 'A': '#2ca02c', 'AM': '#9467bd'}
EYE, HEAD = '#17becf', '#d62728'
EYE_DARK, HEAD_DARK = '#0b6e75', '#7b1113'
EYE_GAIN = '#1f4e79'
UP, DOWN = '#e377c2', '#bcbd22'
SHUF = '#b9b8b2'
SHUF_DARK = '#7f7e79'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
REAL = '#52514e'
HATCH = '///'
AREA_DARK = {'V1': '#0f3c5c', 'RL': '#8a4508', 'A': '#185918', 'AM': '#4f3769'}
LS_LOW, LS_HIGH = '--', ':'
VISIBLE = (-160.0, 20.0)
VIS_SHADE = '#fbf3d5'
R2 = r'$\mathregular{R^2}$'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': FONT, 'axes.labelsize': LABEL, 'xtick.labelsize': LABEL, 'ytick.labelsize': LABEL,
    'legend.fontsize': FONT, 'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.columnspacing': 1.0,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.major.size': 1.7, 'ytick.major.size': 1.7, 'xtick.major.width': 0.45, 'ytick.major.width': 0.45,
    'axes.linewidth': 0.45, 'hatch.linewidth': 0.57, 'pdf.fonttype': 42,
    'figure.constrained_layout.h_pad': 0.05, 'figure.constrained_layout.w_pad': 0.05,
})


def load(path=DATA):
    def walk(g):
        out = {}
        for k, v in g.items():
            if isinstance(v, h5py.Group):
                out[k] = walk(v)
            else:
                x = v[()]
                out[k] = x.decode() if isinstance(x, bytes) else x
        return out
    with h5py.File(path, 'r') as f:
        return walk(f)


def letter(ax, text, x=-39.67, y=None, level=None):
    leg = ax.get_legend()
    if level is not None:
        ax.annotate(text, xy=(0, 1), xycoords=('axes fraction', level), xytext=(x, 0 if y is None else y),
                    textcoords='offset points', ha='right', va='top', fontsize=LETTER, fontweight='bold')
    elif leg is not None and getattr(leg, '_above', False):
        ax.annotate(text, xy=(0, 1), xycoords=leg, xytext=(x, 0 if y is None else y), textcoords='offset points',
                    ha='right', va='top', fontsize=LETTER, fontweight='bold')
    else:
        ax.annotate(text, xy=(0, 1), xycoords='axes fraction', xytext=(x, 4.53 if y is None else y),
                    textcoords='offset points', ha='right', va='bottom', fontsize=LETTER, fontweight='bold')


def legend_above(ax, handles, labels, ncol, y=1.02, **kw):
    leg = ax.legend(handles, labels, ncol=ncol, loc='lower left', bbox_to_anchor=(0, y), borderaxespad=0, **kw)
    leg.set_in_layout(False)
    leg._above = True
    rows = -(-len(handles) // ncol) + (1 if kw.get('title') else 0)
    ax.text(0, y, '\n' * int(np.ceil(1.3 * rows)) + ' ', transform=ax.transAxes, va='bottom', ha='left', fontsize=FONT)
    return leg


def area_ticks(ax, rotate=0):
    ax.set_xticks(range(len(AREAS)))
    ax.set_xticklabels(AREAS, rotation=rotate)


def errbar(ax, x, lo, hi, lw=ERR_LW):
    ax.plot([x, x], [lo, hi], color=INK, lw=lw, zorder=4)


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'



def bar_widths(fig):
    """ Median bar width of every bar plot in the figure, in inches on the page. Axes flagged ax._hist are skipped. """
    from matplotlib.container import BarContainer
    fig.canvas.draw()
    out = {}
    for ax in fig.axes:
        if getattr(ax, '_hist', False):
            continue
        w = [p.get_window_extent().width for c in ax.containers if isinstance(c, BarContainer) for p in c.patches]
        if w:
            out[ax] = np.median(w) / fig.dpi * SCALE
    return out


def shrink_pdf(src, dst, w_in, h_in):
    """ Scale a PDF page to w_in x h_in inches with ghostscript (vectors and fonts kept, images not resampled). """
    import shutil
    import subprocess
    gs = shutil.which('gs')
    if gs is None:
        os.replace(src, dst)
        print('  ghostscript not found: {} left at layout size'.format(dst))
        return
    subprocess.run([gs, '-q', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.5',
                    '-dFIXEDMEDIA', '-dPDFFitPage', '-dAutoRotatePages=/None',
                    '-dDEVICEWIDTHPOINTS={:.2f}'.format(72 * w_in), '-dDEVICEHEIGHTPOINTS={:.2f}'.format(72 * h_in),
                    '-dDownsampleColorImages=false', '-dDownsampleGrayImages=false', '-dDownsampleMonoImages=false',
                    '-dAutoFilterColorImages=false', '-dColorImageFilter=/FlateEncode',
                    '-dAutoFilterGrayImages=false', '-dGrayImageFilter=/FlateEncode',
                    '-o', dst, src], check=True)
    os.remove(src)


def save(fig, name):
    """ Save at page size: the WIDTH-wide layout is shrunk by SCALE, so every size keeps its proportion. """
    W, H = fig.get_size_inches()
    if H > MAX_HEIGHT + 1e-6:
        print('  WARNING: {} is {:.2f} in tall on the page (limit {:.1f})'.format(name, H * SCALE, OUT_MAX_HEIGHT))
    bw = bar_widths(fig)
    if bw:
        print('  bar widths on the page: {:.3f}-{:.3f} in (target {:.3f})'.format(min(bw.values()), max(bw.values()),
                                                                                 BAR * SCALE))
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, name)
    if 'png' in FORMATS:
        fig.savefig(base + '.png', dpi=DPI * SCALE, pil_kwargs={'dpi': (DPI, DPI)})
    if 'pdf' in FORMATS:
        fig.savefig(base + '_layout.pdf', dpi=DPI)
        shrink_pdf(base + '_layout.pdf', base + '.pdf', W * SCALE, H * SCALE)
    plt.close(fig)
    print('saved {} ({:.2f} x {:.2f} in)'.format(base, W * SCALE, H * SCALE))


def visible_gap(ax, x, ys):
    ok = np.any([np.isfinite(y) for y in ys], 0)
    fin = np.flatnonzero(ok)
    for a, b in zip(fin[:-1], fin[1:]):
        if b > a + 1:
            ax.axvspan(x[a], x[b], color=VIS_SHADE, lw=0, zorder=0)


FSI_ROOTS = (os.environ['FM2P_ROOT'].split(':') if os.environ.get('FM2P_ROOT') else
             [os.path.join('/home/dylan/Storage/freely_moving_data/V1PPC', c)
              for c in ('cohort01_recordings', 'cohort02_recordings')])
FSI_CACHE = os.path.join(HERE, 'manuscript_figures', 'field_shift_index.h5')
FSI_MIN_OVERLAP = 120


def _fsi_upsample(F):
    nb, C = F.shape
    x = np.arange(nb) * 360.0 / nb
    xx = np.arange(360.0)
    out = np.full((360, C), np.nan)
    for c in range(C):
        f = F[:, c]
        ok = np.isfinite(f)
        if ok.sum() < nb // 3:
            continue
        out[:, c] = np.interp(xx, np.r_[x[ok] - 360, x[ok], x[ok] + 360], np.r_[f[ok], f[ok], f[ok]])
        gap = np.interp(xx, np.r_[x - 360, x, x + 360], np.r_[~ok, ~ok, ~ok].astype(float))
        out[gap > 0.5, c] = np.nan
    return out


def _fsi_corr(A, B):
    m = np.isfinite(A) & np.isfinite(B)
    n = m.sum(0)
    ma = np.where(m, A, 0).sum(0) / np.maximum(n, 1)
    mb = np.where(m, B, 0).sum(0) / np.maximum(n, 1)
    da, db = np.where(m, A - ma, 0), np.where(m, B - mb, 0)
    den = np.sqrt((da ** 2).sum(0) * (db ** 2).sum(0))
    r = (da * db).sum(0) / np.where(den > 0, den, np.nan)
    r[n < FSI_MIN_OVERLAP] = np.nan
    return r


def field_shift_index(F_low, F_high, delta):

    A, B = _fsi_upsample(F_low), _fsi_upsample(F_high)
    r_ret = _fsi_corr(A, B)
    r_head = _fsi_corr(A, np.roll(B, int(round(delta)), axis=0))
    d_ret, d_head = 1 - r_ret, 1 - r_head
    return (d_head - d_ret) / (d_head + d_ret), r_ret, r_head


def _fsi_compute(ratio_check=None):
    import glob
    import re
    S = lambda x: x.decode() if isinstance(x, (bytes, np.bytes_)) else x
    best = {}
    for root in FSI_ROOTS:
        for p in sorted(glob.glob(os.path.join(root, '**', '*_neural.h5'), recursive=True)):
            if re.search(r'_neural_[^/]+\.h5$', p):
                continue
            sess = os.path.basename(os.path.dirname(os.path.dirname(p)))
            m = re.search(r'(DMM\d+)_(pos\d+)', sess)
            key = (m.group(1), m.group(2)) if m else ('?', '?')
            with h5py.File(p, 'r') as f:
                usable = float(S(f['meta']['n_frames'][()])) * float(S(f['meta']['use_frac'][()]))
            if key not in best or usable > best[key][0]:
                best[key] = (usable, p, sess)
    out = {a: {k: [] for k in ('index', 'r_ret', 'r_head', 'ratio', 'session', 'cell')} for a in AREAS}
    for key in sorted(best):
        _, p, sess = best[key]
        with h5py.File(p, 'r') as f:
            area = np.array(S(f['cells']['area'][()]).split(','))
            es = f['refframe']['eye_shift']
            F_low, F_high, delta = es['F_low'][()], es['F_high'][()], float(es['delta'][()])
            rel, ratio = es['reliable'][()].astype(bool), es['ratio'][()]
        idx, r_ret, r_head = field_shift_index(F_low, F_high, delta)
        for a in AREAS:
            sel = np.flatnonzero((area == a) & rel & np.isfinite(ratio))
            for k, v in (('index', idx), ('r_ret', r_ret), ('r_head', r_head), ('ratio', ratio)):
                out[a][k] += list(v[sel])
            out[a]['session'] += [sess] * len(sel)
            out[a]['cell'] += list(sel)
    for a in AREAS:
        out[a] = {k: np.array(v) for k, v in out[a].items()}
        if ratio_check is not None:
            old, new = np.sort(ratio_check[a]), np.sort(out[a]['ratio'])
            if len(old) != len(new) or not np.allclose(old, new):
                print('WARNING: field-shift index cells for {} ({}) differ from figure_data.h5 ({}).'.format(
                    a, len(new), len(old)))
    return out


def field_shift_cells(ratio_check=None):

    if os.path.isfile(FSI_CACHE):
        with h5py.File(FSI_CACHE, 'r') as f:
            return {a: {k: (v[()].astype(str) if k == 'session' else v[()]) for k, v in f[a].items()}
                    for a in AREAS}
    out = _fsi_compute(ratio_check)
    os.makedirs(os.path.dirname(FSI_CACHE), exist_ok=True)
    with h5py.File(FSI_CACHE, 'w') as f:
        for a, d in out.items():
            g = f.create_group(a)
            for k, v in d.items():
                g[k] = v.astype('S') if k == 'session' else v
    print('saved', FSI_CACHE)
    return out


# ---------------------------------------------------------------------------
# Main Figures 1-5. Each is laid out WIDTH (8.5 in) wide and shrunk to
# OUT_WIDTH (6.5 in) when saved (save()), so text keeps its size relative to
# the figure. Bar plots sit in grid columns whose widths are set so that every
# bar comes out BAR wide; the last column of such a row is an empty spacer that
# takes up the rest of the width. save() prints the bar widths it finds.
# One function per panel below, then one function per figure that places them.
# ---------------------------------------------------------------------------

SP = 1.3       # spacing between the three cell groups of Figure 1F-G (room for the 'not tuned' tick label)

# Grid column widths (layout inches, before shrinking) that make every bar BAR wide. The last entry of
# each list is the empty spacer at the end of the row; save() prints the bar widths it gets.
F1_BOTTOM = [2.885, 1.188, 1.347, 0.66]          # Fig 1 E, F, G, spacer
F2_PANEL, F2_SPACE = 1.474, 0.71               # Fig 2 A-B: four columns, spacer
F2_BOTTOM = [3.2, 3.2]                         # Fig 2 C, D, spacer
F3_TOP = [1.498, 1.498, 3.0, 0.937]             # Fig 3 A, B, C (a 2 x 2 block), spacer
F3_BOTTOM = [2.888, 0.28, 1.50, 1.50]           # Fig 3 D, gap (room for D's legend), E, F
F4_TOP = [1.1, 1.1, 1.5, 1.5, 0.1]              # Fig 4 A (visible, never visible), B, C, spacer


def eye_position_hist(ax, d):
    """ Fig 1A (was 1B): eye-in-head position over frames with the pillar in view, with the eye-third cuts. """
    ax._hist = True
    e = d['B']['bin_edges']
    ax.bar(e[:-1] + np.diff(e) / 2, d['B']['frac_frames'], width=np.diff(e), color='#8a8986')
    # ax.hist(d['B']['frac_frames'], color='#8a8986', bins=30)
    for v in d['B']['third_cuts']:
        ax.axvline(v, color=INK, lw=1.36, ls='--')
    ax.set_xlim(e[0], e[-1])
    ax.set_xlabel('eye in head (deg)')
    ax.set_ylabel('frac. of frames')


def eye_gap(ax, d):
    """ Fig 1B (was 1C): per session, the gap between the low and high eye thirds. """
    g = d['C']['gap_deg']
    # ax.scatter(np.random.default_rng(4).uniform(-0.15, 0.15, len(g)), g, s=6.42, color=INK2, lw=0, zorder=3)
    ax.hist(g, color='#8a8986', bins=10)
    ax.set_xlim([0, 35])
    # ax.plot([-0.3, 0.3], [np.median(g)] * 2, color=INK, lw=1.42)
    # ax.axhline(d['C']['min_gap'], color=INK2, lw=0.57, ls='--')
    # ax.set_xlim(-0.6, 0.6)
    # ax.set_xticks([])
    # ax.set_ylim(bottom=0)
    ax.set_xlabel('gap (deg)')
    ax.set_ylabel('# recordings')


def example_fields(fig, row_ret, row_head, d):
    """ Fig 1C-D (was 1D-E): cells 1-4 at low / high eye third, over retinal (top) and head-centered (bottom) azimuth. """
    first = []
    for j, a in enumerate(AREAS):
        c = d['DE'][a]
        grid, lo, hi, dl = c['grid'], c['low'], c['high'], c['delta']
        ax = fig.add_subplot(row_ret[j])
        ax.set_box_aspect(3 / 4)
        ax.set_title('cell {}'.format(j + 1), fontsize=FONT, pad=4)
        ax.axvspan(*VISIBLE, color=VIS_SHADE, lw=0)
        ax.plot(grid, lo, color=AREA_COLOR[a], lw=1.59, ls=LS_LOW)
        ax.plot(grid, hi, color=AREA_COLOR[a], lw=1.59, ls=LS_HIGH)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel('retinal azimuth (deg)')
        if j == 0:
            ax.set_ylabel('rate')
            legend_above(ax, [Line2D([], [], color=INK, lw=1.59, ls=LS_LOW), Line2D([], [], color=INK, lw=1.59, ls=LS_HIGH)],
                         ['eye right (low third)', 'eye left (high third)'], 2, y=1.15)
            first.append(ax)
        ax = fig.add_subplot(row_head[j])
        ax.set_box_aspect(3 / 4)
        for x, y, ls in (((grid - dl / 2 + 180) % 360 - 180, lo, LS_LOW), ((grid + dl / 2 + 180) % 360 - 180, hi, LS_HIGH)):
            o = np.argsort(x)
            ax.plot(x[o], y[o], color=AREA_COLOR[a], lw=1.59, ls=ls)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel('head-centered\nazimuth (deg)')
        if j == 0:
            ax.set_ylabel('rate')
            first.append(ax)
    return first


def frame_calls(ax, d):
    """ Fig 1E (was 1G): fraction of cells fit better in eye vs head coordinates, light and dark. """
    cols = [('gm_eye_better', EYE), ('gm_eye_better_dark', EYE_DARK), ('gm_head_better', HEAD),
            ('gm_head_better_dark', HEAD_DARK)]
    w = 0.2
    rng = np.random.default_rng(2)
    for i, a in enumerate(AREAS):
        for j, (col, color) in enumerate(cols):
            v = d['G'][a][col]
            x = i + (j - 1.5) * w
            ax.bar(x, v['frac'], width=w * 0.9, color=color)
            s = v['sessions']
            ax.scatter(x + rng.uniform(-0.045, 0.045, len(s)), s, s=3.85, color=INK, lw=0, zorder=3)
            errbar(ax, x, v['lo'], v['hi'])
    area_ticks(ax)
    ax.set_ylabel('frac. cells')
    ax.set_ylim(0, 1.05 * max(d['G'][a][c]['hi'] for a in AREAS for c, _ in cols))
    return legend_above(ax, [Patch(color=c) for _, c in cols],
                        ['eye (light)', 'eye (dark)', 'head (light)', 'head (dark)'], 2)


def scale_vs_shift(axs, d):
    """ Fig 2A (was 2C): per gain cell, |r| of the eye gain with the field (scale) vs with its slope (shift). """
    for j, (ax, a) in enumerate(zip(axs, AREAS)):
        c = d['C'][a]
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.45, ls='--')
        ax.scatter(c['shift_r'], c['scale_r'], s=12.84, color=AREA_COLOR[a], alpha=0.7, lw=0)
        ax.scatter([np.median(c['shift_r'])], [np.median(c['scale_r'])], s=96.33, marker='+', color=INK, lw=1.42, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_aspect('equal')
        ax.set_xlabel('|r| shift')
        if j == 0:
            ax.set_ylabel('|r| scale')


def gain_positions(axs, d):
    """ Fig 2B (was 2D): preferred retinal azimuth of gain cells, gain up / down with nasal eye, against chance. """
    ymax = 0
    for j, (ax, a) in enumerate(zip(axs, AREAS)):
        c = d['D'][a]
        e = c['bin_edges']
        cen, bw = e[:-1] + np.diff(e) / 2, np.diff(e)[0]
        n = max((c['gain_up'] + c['gain_down']).sum(), 1)
        up, dn = c['gain_up'] / n, c['gain_down'] / n
        ax.axvspan(*VISIBLE, color=VIS_SHADE, lw=0)
        ax.bar(cen, up, width=bw * 0.85, color=UP)
        ax.bar(cen, dn, bottom=up, width=bw * 0.85, color=DOWN)
        ax.step(e, np.r_[c['chance'], c['chance'][-1]] / n, where='post', color=INK, lw=0.57, ls='--', zorder=3)
        # for k in np.flatnonzero(c['p'] < 0.05):
        #     ax.text(cen[k], up[k] + dn[k], '*', ha='center', va='bottom', fontsize=2 * FONT)
        ymax = max(ymax, (up + dn).max())
        ax.set_xlim(e[0], e[-1])
        ax.set_xticks([-90, 0, 90, 180])
        ax.set_xlabel('pref. azimuth (deg)')
        if j == 0:
            ax.set_ylabel('frac. gain cells')
        else:
            ax.tick_params(labelleft=False)
    for ax in axs:
        ax.set_ylim(0, 1.2 * ymax)
    return legend_above(axs[0], [Line2D([], [], color=INK, lw=0.57, ls='--'), Patch(color=UP), Patch(color=DOWN)],
                        ['chance', 'gain up with nasal eye', 'gain down with nasal eye'], 3)


SIM_BARS = [('real', REAL), ('sim_eye_gain', EYE_GAIN), ('sim_head', HEAD)]


def sim_calls(ax, d, panel):
    """ Fig 2C-D (was 3A-B): eye- (A) or head-centered (B) calls, real cells vs simulated cells of known frame. """
    w = 0.26
    hi = 0
    for i, a in enumerate(AREAS):
        for j, (key, color) in enumerate(SIM_BARS):
            v = d[panel][a][key]
            x = i + (j - 1) * w
            sim = key != 'real'
            ax.bar(x, v['frac'], width=w * 0.9, color=color, hatch=HATCH if sim else None,
                   edgecolor=INK2 if sim else 'none', lw=0.28)
            errbar(ax, x, v['lo'], v['hi'])
            hi = max(hi, v['hi'])
    area_ticks(ax)
    return hi


def area_pairs(ax, get, rng=None):
    """ Per area, real (area colour) and null (grey) bars; get(area, kind) -> (median, lo, hi, dots or None). """
    hi = 0
    for i, a in enumerate(AREAS):
        for kind, x, color in (('real', i - 0.18, AREA_COLOR[a]), ('null', i + 0.18, SHUF)):
            m, lo, h, dots = get(a, kind)
            ax.bar(x, m, width=0.34, color=color)
            errbar(ax, x, lo, h)
            if dots is not None:
                ax.scatter(x + rng.uniform(-0.07, 0.07, len(dots)), dots, s=3.85, color=INK, lw=0, zorder=3)
                h = np.max(dots)
            hi = max(hi, h)
    area_ticks(ax)
    ax.set_ylim(0, 1.05 * hi)


def readout_by_direction(axs, d):
    """ Fig 3C (was 4E): readout R2 above the shuffled-eye null by head-centered direction, 141 cells per area. """
    for j, (ax, a) in enumerate(zip(axs, AREAS)):
        c = d['E'][a]
        ax.axvspan(*VISIBLE, color=VIS_SHADE, lw=0)
        x = np.r_[c['direction'], c['direction'][0] + 360]
        wrap = lambda y: np.r_[y, y[0]]
        ax.fill_between(x, wrap(c['p10']), wrap(c['p90']), color=AREA_COLOR[a], alpha=0.25, lw=0)
        ax.plot(x, wrap(c['median']), color=AREA_COLOR[a], lw=1.98, marker='o', ms=3.97)
        ax.axhline(0, color=INK2, lw=0.45)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, 0, 180])
        if j >= 2:
            ax.set_xlabel('head-centered\ndirection (deg)')
        else:
            ax.tick_params(labelbottom=False)
        if j % 2 == 0:
            ax.set_ylabel('real - null ' + R2)
    lo = min(ax.get_ylim()[0] for ax in axs)
    hi = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(lo, hi)


def hb_gain(ax, d):
    """ Fig 3D (was 5A): head-on-body gain on head-centered fields, real and shuffled, alone and beyond turning / speed. """
    hi = 0
    for i, a in enumerate(AREAS):
        for j, col in enumerate(('call', 'beyond_turn')):
            x = i + (2 * j - 1.5) * 0.2
            for xx, key, color in ((x, col, (HEAD, HEAD_DARK)[j]), (x + 0.2, col + '_shuffled', (SHUF, SHUF_DARK)[j])):
                v = d['A'][a][key]
                ax.bar(xx, v['frac'], width=0.18, color=color)
                errbar(ax, xx, v['lo'], v['hi'])
                hi = max(hi, v['hi'])
    area_ticks(ax)
    ax.set_ylim(0, 1.05 * hi)
    ax.set_ylabel('frac. of head-centered-\ntuned cells')
    return legend_above(ax, [Patch(color=HEAD), Patch(color=HEAD_DARK), Patch(color=SHUF), Patch(color=SHUF_DARK)],
                        ['hb gain', 'hb gain beyond turning / speed', 'shuffled', 'shuffled'], 2)


GAIN_ROWS = (('visible', 'visible'), ('never_visible', 'never visible'))
COMP_LS, COMP_LW = ('-', '--', ':'), 1.98


def gain_profiles(ax, azimuth, curves, color, never):
    """ One panel of Fig 4A-B (was 4A, 4G): three leading gain components over retinal azimuth. """
    for k in range(3):
        ax.plot(azimuth, curves[k], color=color, ls=COMP_LS[k], lw=COMP_LW)
    ax.axhline(0, color=INK2, lw=0.34)
    if never:
        visible_gap(ax, azimuth, curves)
    y = np.concatenate([c[np.isfinite(c)] for c in curves])
    ax.set_ylim(min(0.0, 1.1 * y.min()), 1.1 * y.max())
    ax.set_xlabel('retinal azimuth\n(deg)')


def vis_never_bars(ax, d, key):
    """ Fig 4C-D (was 4B-C): per area, the visible-field (area colour) and never-visible (dark) value, all tuned cells. """
    hi = 0
    for i, a in enumerate(AREAS):
        for (row, _), x, color in zip(GAIN_ROWS, (i - 0.18, i + 0.18), (AREA_COLOR[a], AREA_DARK[a])):
            v = d['BCD'][row][a][key]
            ax.bar(x, v['value'], width=0.34, color=color)
            errbar(ax, x, v['lo'], v['hi'])
            hi = max(hi, v['hi'], v['value'])
    area_ticks(ax)
    ax.set_ylim(0, 1.1 * hi)


def figure1(D):
    """ Pillar fields are eye-centered and visual. Old panels 1B, 1C, 1D, 1E, 1G, 9A, 9B. """
    d = D['fig1']
    fig = plt.figure(figsize=(WIDTH, 9.3), layout='constrained')
    gs = fig.add_gridspec(4, 1, height_ratios=[2.3, 2.05, 2.05, 2.9])
    top = gs[0].subgridspec(1, 2, width_ratios=[4.2, 2.95])
    row_c = gs[1].subgridspec(1, 4)
    row_d = gs[2].subgridspec(1, 4)
    bottom = gs[3].subgridspec(1, 4, width_ratios=F1_BOTTOM)

    ax = fig.add_subplot(top[0])
    eye_position_hist(ax, d)
    letter(ax, 'A')
    ax = fig.add_subplot(top[1])
    eye_gap(ax, d)
    letter(ax, 'B')#, x=-34)

    axC, axD = example_fields(fig, row_c, row_d, d)
    letter(axC, 'C')
    letter(axD, 'D')

    ax = fig.add_subplot(bottom[0])
    leg = frame_calls(ax, d)
    letter(ax, 'E')
    axF, axG = fig.add_subplot(bottom[1]), fig.add_subplot(bottom[2])
    dark_panels(axF, axG)
    letter(axF, 'F', level=leg)
    letter(axG, 'G', level=leg)
    save(fig, 'Figure1')


def figure2(D):
    """ Eye position scales the retinal field, and that gain explains the head-centered calls. Old 2C, 2D, 3A, 3B. """
    fig = plt.figure(figsize=(WIDTH, 7.4), layout='constrained')
    gs = fig.add_gridspec(2, 1, height_ratios=[4.3, 3.1])
    upper = gs[0].subgridspec(2, 5, width_ratios=[F2_PANEL] * 4 + [F2_SPACE], height_ratios=[2.0, 2.3])
    bottom = gs[1].subgridspec(1, 2, width_ratios=F2_BOTTOM)

    axs = [fig.add_subplot(upper[0, j]) for j in range(4)]
    scale_vs_shift(axs, D['fig2'])
    letter(axs[0], 'A')
    axs = [fig.add_subplot(upper[1, j]) for j in range(4)]
    gain_positions(axs, D['fig2'])
    letter(axs[0], 'B')

    d = D['fig3']
    axC = fig.add_subplot(bottom[0])
    axD = fig.add_subplot(bottom[1], sharey=axC)
    hi = max(sim_calls(axC, d, 'A'), sim_calls(axD, d, 'B'))
    axC.set_ylim(0, 1.05 * hi)
    axC.set_ylabel('frac. tuned cells')
    axD.tick_params(labelleft=True)
    leg = legend_above(axC, [Patch(facecolor=c, hatch=HATCH if k != 'real' else None, edgecolor=INK2 if k != 'real' else 'none')
                             for k, c in SIM_BARS], ['real', 'simulated eye-centered + gain', 'simulated head-centered'], 1)
    letter(axC, 'C')
    letter(axD, 'D', level=leg)
    save(fig, 'Figure2')


def figure3(D):
    """ A linear readout of the gain fields recovers head- and body-centered direction. Old 3C, 3D, 4E, 5A, 5B, 5C. """
    fig = plt.figure(figsize=(WIDTH, 6.6), layout='constrained')
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 3.3])
    top = gs[0].subgridspec(1, 4, width_ratios=F3_TOP)
    block = top[2].subgridspec(2, 2)
    bottom = gs[1].subgridspec(1, 4, width_ratios=F3_BOTTOM)

    d = D['fig3']
    axA, axB = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    for ax, panel, ylab in ((axA, 'C', 'readout ' + R2), (axB, 'D', 'per-cell signal')):
        area_pairs(ax, lambda a, k: (d[panel][a][k]['median'], d[panel][a][k]['p10'], d[panel][a][k]['p90'], None))
        ax.set_ylabel(ylab)
    axB.text(0, 1.02, '1e−3', transform=axB.transAxes, ha='left', va='bottom', fontsize=FONT)
    leg = legend_above(axA, [Patch(color=INK2), Patch(color=SHUF)], ['real', 'shuffled eye'], 1, y=1.1)
    letter(axA, 'A')
    letter(axB, 'B', level=leg)
    axs = [fig.add_subplot(block[j // 2, j % 2]) for j in range(4)]
    readout_by_direction(axs, D['fig4'])
    letter(axs[0], 'C', x=-48, level=leg)

    d = D['fig5']
    ax = fig.add_subplot(bottom[0])
    leg = hb_gain(ax, d)
    letter(ax, 'D')
    rng = np.random.default_rng(3)
    axE, axF = fig.add_subplot(bottom[2]), fig.add_subplot(bottom[3])
    area_pairs(axE, lambda a, k: (d['B'][a][k]['median'], d['B'][a][k]['p10'], d['B'][a][k]['p90'], d['B'][a][k]['values']),
               rng)
    axE.set_ylabel('per-cell signal')
    axE.text(0, 1.02, '1e-3', transform=axE.transAxes, ha='left', va='bottom', fontsize=FONT)
    # F: the same pooled body-specific readout as E, as R2 at a cell count every area has (matched, as A).
    r2 = body_readout_r2()
    area_pairs(axF, lambda a, k: (np.median(r2[a][k]), np.percentile(r2[a][k], 10), np.percentile(r2[a][k], 90), r2[a][k]),
               rng)
    axF.set_ylabel('readout ' + R2)
    legend_above(axE, [Patch(color=INK2), Patch(color=SHUF)], ['real', 'hb shuffled'], 1, y=1.1)
    letter(axE, 'E', level=leg)
    letter(axF, 'F', level=leg)
    save(fig, 'Figure3')


def figure4(D):
    """ Every area has the same gain components; A and AM carry the most signal. Old 4A, 4B, 4C, 4G. """
    d = D['fig4']
    fig = plt.figure(figsize=(WIDTH, 6.5), layout='constrained')
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 4.0])
    top = gs[0].subgridspec(1, 5, width_ratios=F4_TOP)
    bottom = gs[1].subgridspec(2, 4)

    for c, (key, name) in enumerate(GAIN_ROWS):
        ag = d['AG'][key]
        ax = fig.add_subplot(top[c])
        gain_profiles(ax, ag['azimuth'], ag['needed'], INK, key == 'never_visible')
        ax.set_ylabel('{}:\ngain profile'.format(name))
        if c == 0:
            leg = legend_above(ax, [Line2D([], [], color=INK, ls=ls, lw=COMP_LW) for ls in COMP_LS], ['1', '2', '3'],
                               ncol=3, title='gain component', handlelength=1.6, handletextpad=0.4, columnspacing=0.8,
                               alignment='left')
            letter(ax, 'A')

    axB, axC = fig.add_subplot(top[2]), fig.add_subplot(top[3])
    vis_never_bars(axB, d, 'need')
    axB.set_ylabel('gain signal (x100)')
    vis_never_bars(axC, d, 'coverage')
    axC.set_ylabel('coverage')
    legend_above(axB, [Patch(color='#9a9a9a'), Patch(color='#3a3a3a')], ['visible', 'never visible'], 1)
    letter(axB, 'B', level=leg)
    letter(axC, 'C', level=leg)

    for r, (key, name) in enumerate(GAIN_ROWS):
        ag = d['AG'][key]
        for j, a in enumerate(AREAS):
            ax = fig.add_subplot(bottom[r, j])
            gain_profiles(ax, ag['azimuth'], ag[a], AREA_COLOR[a], key == 'never_visible')
            if j == 0:
                ax.set_ylabel('{}:\ngain profile'.format(name))
                if r == 0:
                    letter(ax, 'D')
    save(fig, 'Figure4')


# ---------------------------------------------------------------------------
# EBC figures (old Figures 6-8), unchanged apart from their file names.
# ---------------------------------------------------------------------------


def figure6(D):
    d = D['fig6']
    fig = plt.figure(figsize=(WIDTH, 9.66), layout='constrained')
    gs = fig.add_gridspec(3, 1, height_ratios=[3.57, 3.15, 2.94])
    row_a = gs[0].subgridspec(1, 2)
    row_b = gs[1].subgridspec(2, 7, width_ratios=[1, 1, 1, 0.12, 1, 1, 1])
    row_c = gs[2].subgridspec(2, 7, width_ratios=[1, 1, 1, 0.12, 1, 1, 1])
    src_style = (('real', REAL, None), ('eye_anchored', EYE, HATCH), ('head_anchored', HEAD, HATCH))

    axs = [fig.add_subplot(row_a[0])]
    axs.append(fig.add_subplot(row_a[1], sharey=axs[0]))
    for ax in axs:
        ax.set_box_aspect(3 / 4)
    hi = 0
    for ax, crit, title in zip(axs, ('ebc', 'rbc'), ('pass as EBC', 'pass as RBC')):
        w = 0.26
        for i, a in enumerate(AREAS):
            for j, (src, color, hatch) in enumerate(src_style):
                v = d['A'][crit][a][src]
                x = i + (j - 1) * w
                ax.bar(x, v['frac'], width=w * 0.9, color=color, hatch=hatch, edgecolor=INK2 if hatch else 'none', lw=0.28)
                errbar(ax, x, v['lo'], v['hi'])
                hi = max(hi, v['hi'])
        area_ticks(ax)
        ax.set_title(title, fontsize=LABEL, pad=28.33)
    axs[0].set_ylim(0, 1.05 * hi)
    axs[0].set_ylabel('frac. of all cells')
    legend_above(axs[0], [Patch(color=REAL), Patch(facecolor=EYE, edgecolor=INK2, hatch=HATCH),
                          Patch(facecolor=HEAD, edgecolor=INK2, hatch=HATCH)],
                 ['real cells', 'simulated eye-anchored', 'simulated head-anchored'], 3)
    letter(axs[0], 'A')

    mb = d['map_bins']
    step, dmax = float(mb['angle_step_deg']), float(mb['dist_max_cm'])
    th = np.radians(np.r_[np.arange(0.0, 360.0, step), 360.0] - step / 2)
    rr = np.arange(0.0, dmax + float(mb['dist_step_cm']) / 2, float(mb['dist_step_cm']))
    ang = np.arange(0.0, 360.0, step)

    def mean_dir(M):
        t = np.radians(ang)[:, None]
        return np.angle((np.nan_to_num(M) * np.exp(1j * t)).sum())

    for h, name in enumerate(('cell9', 'cell10')):
        cell = d['BC'][name]
        maps = cell['maps']
        col0 = 4 * h
        for j, frame in enumerate(('head', 'gaze')):
            for k in range(3):
                M = maps['real'][frame][k]
                ax = fig.add_subplot(row_b[j, col0 + k], projection='polar')
                ax.pcolormesh(th, rr, M.T, cmap='magma', vmin=np.nanmin(M), vmax=np.nanmax(M), shading='flat')
                ax.set_theta_zero_location('N')
                ax.set_xticks(np.radians([90, 270]))
                ax.set_xticklabels(['L', 'R'], fontsize=FONT)
                if j == 0:
                    ax.set_title(('all frames', 'eye temporal', 'eye nasal')[k], fontsize=FONT, pad=4)
                    if k == 1:
                        ax.text(0.5, 1.32, 'cell {}'.format(9 + h), transform=ax.transAxes, ha='center', va='bottom',
                                fontsize=LABEL, fontweight='bold')
                ax.tick_params(pad=-1.13)
                ax.set_yticks([])
                ax.grid(False)
                ax.set_ylim(0, dmax)
                if k > 0:
                    t_all = mean_dir(maps['real'][frame][0])
                    ax.annotate('', xy=(t_all, 0.85 * dmax), xytext=(t_all, 0.0),
                                arrowprops=dict(arrowstyle='-|>', color='white', lw=1.42, ls='--', alpha=0.5,
                                                mutation_scale=12.47, shrinkA=0, shrinkB=0))
                t0 = mean_dir(M)
                ax.annotate('', xy=(t0, 0.85 * dmax), xytext=(t0, 0.0),
                            arrowprops=dict(arrowstyle='-|>', color='white', lw=1.42, mutation_scale=12.47, shrinkA=0, shrinkB=0))
                if k == 0:
                    ax.text(-0.3, 0.5, '{} frame'.format(frame), transform=ax.transAxes, rotation=90, ha='center',
                            va='center', fontsize=LABEL)
                if h == 0 and j == 0 and k == 0:
                    letter(ax, 'B', x=-34)
            for i, lab in enumerate(('real', 'pure_rbc', 'pure_ebc')):
                W, L, H = maps[lab][frame]
                ax = fig.add_subplot(row_c[j, col0 + i])
                pk = ang[np.nanargmax(np.nanmean(W, 1))]
                rel = (ang - pk + 180) % 360 - 180
                o = np.argsort(rel)
                ax.plot(rel[o], np.nanmean(L, 1)[o], color=INK, lw=1.59, ls=LS_LOW)
                ax.plot(rel[o], np.nanmean(H, 1)[o], color=INK, lw=1.59, ls=LS_HIGH)
                ax.axvline(0, color=INK2, lw=0.34, ls=':')
                ax.set_xlim(-180, 180)
                ax.set_xticks([-180, 0, 180])
                ax.set_xlabel('wall angle (deg)' if j == 1 else '')
                ax.set_ylabel('{}: rate'.format(frame) if i == 0 else '')
                ax._row = (h, j)
                if j == 0:
                    ax.set_title(('real cell', 'simulated RBC', 'simulated EBC')[i], fontsize=FONT, pad=4)
                if h == 0 and j == 0 and i == 0:
                    legend_above(ax, [Line2D([], [], color=INK, lw=1.59, ls=LS_LOW), Line2D([], [], color=INK, lw=1.59, ls=LS_HIGH)],
                                 ['eye temporal', 'eye nasal'], 2, y=1.3)
                    letter(ax, 'C')
    groups = {}
    for ax in fig.axes:
        if hasattr(ax, '_row'):
            groups.setdefault(ax._row, []).append(ax)
    for axs_ in groups.values():
        hi = max(a.get_ylim()[1] for a in axs_)
        for i, a in enumerate(axs_):
            a.set_ylim(0, hi)
            a.yaxis.set_major_locator(plt.MaxNLocator(2))
            a.tick_params(labelleft=(i == 0))

    save(fig, 'EBC_figure1')


def figure7(D):
    d = D['fig6']
    fig = plt.figure(figsize=(WIDTH, 7.07), layout='constrained')
    gs = fig.add_gridspec(2, 1, height_ratios=[3.68, 3.39])
    row_de = gs[0].subgridspec(1, 2)
    row_fg = gs[1].subgridspec(1, 2)

    ax = fig.add_subplot(row_de[0])
    ax.set_box_aspect(3 / 4)
    bins = np.linspace(-2, 3, 41)
    for key, color, filled in (('pure_ebc_sim', HEAD, False), ('pure_rbc_sim', EYE, False), ('real', REAL, True)):
        v = np.clip(d['D'][key], bins[0], bins[-1])
        ax.hist(v, bins, density=True, histtype='stepfilled' if filled else 'step', color=color,
                alpha=0.35 if filled else 1, lw=1.7)
    for x0 in (0, 1):
        ax.axvline(x0, color=INK2, lw=0.34, ls=':')
    ax.set_xlabel('position between references (0 = EBC, 1 = RBC)')
    ax.set_ylabel('density')
    legend_above(ax, [Line2D([], [], color=HEAD, lw=1.7), Line2D([], [], color=EYE, lw=1.7), Patch(color=REAL, alpha=0.35)],
                 ['pure EBC, simulated', 'pure RBC, simulated', 'real EBCs'], 2)
    letter(ax, 'A')

    def scaled_panel(ax, key, ylab):
        ax.set_box_aspect(3 / 4)
        ax.axhspan(-0.5, 0, color=HEAD, alpha=0.08, lw=0)
        ax.axhspan(1, 1.75, color=EYE, alpha=0.08, lw=0)
        ax.axhline(0, color=HEAD, lw=0.85, ls='--')
        ax.axhline(1, color=EYE, lw=0.85, ls='--')
        ax.axhline(0.5, color=INK, lw=0.85, ls='--', zorder=1)
        names = AREAS + ['all']
        for i, a in enumerate(names):
            if a not in d['scaled']:
                continue
            y, lo, hi = d['scaled'][a][key]
            color = AREA_COLOR.get(a, INK)
            ax.plot([i, i], [lo, hi], color=color, lw=ERR_LW, zorder=3)
            ax.plot(i, y, 'o', color=color, ms=4.5, zorder=4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(-0.5, 1.75)
        ax.set_yticks([0, 0.5, 1, 1.5])
        ax.set_ylabel(ylab)
        ax.text(1.02, 0, 'all pure\nEBCs', transform=ax.get_yaxis_transform(), color=HEAD, va='center', fontsize=FONT)
        ax.text(1.02, 1, 'all pure\nRBCs', transform=ax.get_yaxis_transform(), color=EYE, va='center', fontsize=FONT)

    ax = fig.add_subplot(row_de[1])
    scaled_panel(ax, 'calls', 'single-cell calls\n(0 = EBC, 1 = RBC)')
    letter(ax, 'B')

    ax = fig.add_subplot(row_fg[0])
    ax.set_anchor('W')
    scaled_panel(ax, 'conc', 'gaze \u2212 head concentration\n(0 = EBC, 1 = RBC)')
    letter(ax, 'C')

    save(fig, 'EBC_figure2')


def bracket(ax, i, j, y, p, h=0.04):
    tr = ax.get_xaxis_transform()
    ax.plot([i, i, j, j], [y - h, y, y, y - h], color=INK, lw=0.57, transform=tr, clip_on=False)
    ax.text((i + j) / 2, y, stars(p), ha='center', va='bottom', fontsize=FONT, transform=tr, clip_on=False)


def figure8(D):
    d = D['fig7']
    fig = plt.figure(figsize=(WIDTH, 2.25), layout='constrained')
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 2 / 3, 1, 1])

    ax = axA = fig.add_subplot(gs[0])
    ax.set_box_aspect(3 / 4)
    groups = (('non-EBC cells', SHUF), ('EBC, head-leaning', HEAD), ('EBC, gaze-leaning', EYE))
    vals = []
    for i, (g, color) in enumerate(groups):
        y, lo, hi = d['A'][g]
        ax.bar(i, y, width=0.6, color=color)
        errbar(ax, i, lo, hi)
        vals.append(hi)
    p = d['p']['A']
    bracket(ax, 0, 1, 1.08, p[0])
    bracket(ax, 1, 2, 1.08, p[2])
    bracket(ax, 0, 2, 1.3, p[1])
    ax.set_ylim(0, 1.05 * max(vals))
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(('non-\nEBC', 'head', 'gaze'))
    ax.set_ylabel('sparse-noise RF\n(frac. cells)')
    letter(ax, 'A', x=-46, y=30)

    ax = axB = fig.add_subplot(gs[1])
    ax.set_box_aspect(3 / 4 * 3 / 2)
    vals = []
    for i, (g, color) in enumerate(groups[1:]):
        y, lo, hi = d['BCD'][g]['map_r_head']
        ax.bar(i, y, width=0.6, color=color)
        errbar(ax, i, lo, hi)
        vals.append(hi)
    bracket(ax, 0, 1, 1.08, d['p']['D'][2])
    ax.set_ylim(0, 1.05 * max(vals))
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(('head', 'gaze'))
    ax.set_ylabel('light vs dark\nwall map (r)')
    letter(ax, 'B', x=-46, y=30)

    if 'C' in d:
        ax = fig.add_subplot(gs[2])
        ax.set_box_aspect(3 / 4)
        w = 0.26
        styles = (('real', REAL, None), ('all_pure_ebc', HEAD, HATCH), ('all_pure_rbc', EYE, HATCH))
        for i, key in enumerate(('in_view', 'out_of_view')):
            for j, (src, color, hatch) in enumerate(styles):
                y, lo, hi = d['C'][key][src]
                x = i + (j - 1) * w
                ax.bar(x, y, width=w * 0.9, color=color, hatch=hatch, edgecolor=INK2 if hatch else 'none', lw=0.28)
                errbar(ax, x, lo, hi)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks(range(2))
        ax.set_xticklabels(('field in\nview', 'field out\nof view'))
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel('estimated frac.\nretinocentric')
        legend_above(ax, [Patch(color=REAL), Patch(facecolor=HEAD, edgecolor=INK2, hatch=HATCH),
                          Patch(facecolor=EYE, edgecolor=INK2, hatch=HATCH)],
                     ['real EBCs', 'if all pure EBCs', 'if all pure RBCs'], 1)
        letter(ax, 'C', x=-46)

    W = fig.get_size_inches()[0]
    for _ in range(6):
        fig.canvas.draw()
        wa, wb = axA.get_position().width * W, axB.get_position().width * W
        r = gs.get_width_ratios()
        r[1] *= (2 / 3 * wa) / wb
        gs.set_width_ratios(r)
    save(fig, 'EBC_figure3')


# ---------------------------------------------------------------------------
# Figure 5C and Figures 9-11 read the analysis notebook's own store
# (cross_area_summary_data.h5) and the session _neural.h5 files directly,
# not figure_data.h5. Per-cell values are cached in CELL_CACHE (delete it to
# rebuild, e.g. after rerunning the notebook or new_analysis).
# ---------------------------------------------------------------------------

STORE = os.environ.get('FM2P_STORE', '/home/dylan/Storage/freely_moving_data/V1PPC/cross_area_summary_data.h5')
CELL_CACHE = os.path.join(HERE, 'manuscript_figures', 'cell_table.h5')
ATLAS = os.path.join(os.path.dirname(HERE), 'fm2p', 'utils', 'vfs_contours.json')
BODY_READOUT_N = 50            # largest pool size run for every area (AM has 72 head-centred-tuned cells)
Q_GRID = np.arange(-175.0, 180.0, 10.0)
N_VM, VM_KAPPA = 8, 2.0
MIN_DARK_EYE_FRAMES = 1200     # dark frames with eye tracking needed for the dark pillar models (as notebook O1a)

MAP_STEP = 2.0                 # map pixel, reference-atlas px
MAP_SIGMA, MAP_SIGMA_MAX = 14.0, 40.0   # Gaussian smoothing SD, reference-atlas px: smallest, and largest allowed
MAP_K = 30                     # each pixel's SD widens from MAP_SIGMA until this many cells of the mapped kind are within it
FOOT_MIN_W = 3.0               # imaged footprint: summed MAP_SIGMA kernel weight of ALL cells (same for every panel)
COV_SIGMA, COV_SIGMA_MAX, COV_K, COV_SPLITS = 30.0, 60.0, 100, 20   # local gain-signal / coverage maps
NO_DATA = '#cfcec9'            # imaged, but too few cells of the kind being mapped


def _txt(x):
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else x


def _unpack(g):
    """ One object written by the notebook's h5_pack (notebook cell 'refit-store'). """
    import pandas as pd
    if isinstance(g, h5py.Dataset):
        return _txt(g[()])
    t = _txt(g['__type__'][()])
    sub = lambda k: _unpack(g[k])
    if t == 'none':
        return None
    if t in ('bool', 'int', 'float'):
        return {'bool': bool, 'int': int, 'float': float}[t](g['value'][()])
    if t == 'array0':
        return np.asarray(sub('value')).reshape(())
    if t == 'str_array':
        v, shape = g['value'][()], tuple(g['shape'][()])
        return np.char.decode(v, 'utf-8').reshape(shape) if v.size else np.zeros(shape, str)
    if t == 'object_array':
        items = sub('items')
        a = np.empty(len(items), object)
        a[:] = items
        return a.reshape(tuple(g['shape'][()]))
    if t in ('list', 'tuple', 'set'):
        return {'list': list, 'tuple': tuple, 'set': set}[t](sub(str(i)) for i in range(int(g['n'][()])))
    if t == 'dict':
        return {k: sub(str(i)) for i, k in enumerate(sub('keys'))}
    if t == 'range_index':
        return pd.RangeIndex(int(g['start'][()]), int(g['stop'][()]), int(g['step'][()]))
    if t == 'multi_index':
        return pd.MultiIndex.from_tuples(sub('tuples'))
    if t == 'index':
        return pd.Index(sub('values'))
    if t == 'frame':
        cols, dtypes = list(sub('columns')), sub('dtypes')
        df = pd.DataFrame({c: pd.Series(sub(str(i)), dtype=dt if dt != 'object' else object)
                           for i, (c, dt) in enumerate(zip(cols, dtypes))})
        df.index = sub('index')
        return df
    if t == 'series':
        return pd.Series(sub('values'), index=sub('index'))
    raise ValueError('unknown stored type {}'.format(t))


def store_step(step):
    """ The variables one notebook step saved, e.g. store_step('P')['BODY_POOL']. """
    with h5py.File(STORE, 'r') as f:
        return _unpack(f[step])


def body_readout_r2(n_cells=BODY_READOUT_N):
    """ Pooled body-specific readout R2 (notebook P2, measured noise, alpha 1) at n_cells per area: hosts x seeds. """
    pool = store_step('P')['BODY_POOL']
    out = {a: {'real': [], 'null': []} for a in AREAS}
    for (host, a), r in pool.items():
        if a not in out:
            continue
        zi, ai = list(r['sizes']).index(n_cells), list(r['alphas']).index(1.0)
        for ki, kind in enumerate(('real', 'null')):
            out[a][kind] += list(r['r2'][ki, zi, ai])
    return {a: {k: np.asarray(v, float) for k, v in d.items()} for a, d in out.items()}


def _vm(ang):
    c = np.radians(np.arange(N_VM) * 360.0 / N_VM)
    return np.exp(VM_KAPPA * (np.cos(np.radians(np.asarray(ang, float))[:, None] - c[None, :]) - 1.0))


def gain_bases(cols):
    """ As notebook Q4: basis U of the profiles the model can express on these grid columns, needed shapes P in U. """
    x = Q_GRID[cols]
    U, s_, _ = np.linalg.svd(_vm(x), full_matrices=False)
    U = U[:, s_ > 1e-3 * s_[0]]
    return U, U.T @ ((_vm(x + 1.0) - _vm(x - 1.0)) / 2.0)


def neural_sessions():
    """ One _neural.h5 per animal x position, the one with the most usable frames (the notebook's DEDUPE rule). """
    import glob
    import re
    best = {}
    for root in FSI_ROOTS:
        for p in sorted(glob.glob(os.path.join(root, '**', '*_neural.h5'), recursive=True)):
            if re.search(r'_neural_[^/]+\.h5$', p):
                continue
            sess = os.path.basename(os.path.dirname(os.path.dirname(p)))
            m = re.search(r'(DMM\d+)_(pos\d+)', sess)
            key = (m.group(1), m.group(2)) if m else ('?', sess)
            with h5py.File(p, 'r') as f:
                usable = float(_txt(f['meta']['n_frames'][()])) * float(_txt(f['meta']['use_frac'][()]))
            if key not in best or usable > best[key][0]:
                best[key] = (usable, p, sess, key[0])
    return [best[k][1:] for k in sorted(best)]


def _session_cells(path, sess, animal):
    with h5py.File(path, 'r') as f:
        c, rf, cv, gm = f['cells'], f['refframe'], f['covariates'], f['gain_model']
        area = np.array(_txt(c['area'][()]).split(','))
        C = len(area)
        pos = c['vfs_cell_pos'][()]
        d = {'session': np.array([sess] * C), 'animal': np.array([animal] * C), 'cell': np.arange(C), 'area': area,
             'x': pos[:, 0], 'y': pos[:, 1], 'rate_light': c['mean_rate_light'][()], 'rate_dark': c['mean_rate_dark'][()]}
        for tag, key in (('', 'eye_models'), ('_dark', 'eye_models_dark')):
            for k, name in (('tuned', 'tuned'), ('eye_better', 'eye_centred'), ('head_better', 'head_centred')):
                d[name + tag] = rf[key][k][()].astype(bool) if key in rf else np.zeros(C, bool)
        use, light = cv['use'][()].astype(bool), cv['light'][()].astype(bool)
        rho = cv['rho'][()]
        fin = np.isfinite(rho) & np.isfinite(cv['eta'][()]) & np.isfinite(cv['e'][()])
        dt = float(np.median(np.diff(cv['t'][()])))
        coef, x_sd = gm['coef'][()], gm['x_sd'][()]
        n_nuis = len(_txt(gm['nuis_names'][()]).split(','))
    d['dark_eye_frames'] = np.full(C, int((use & ~light & fin).sum()))
    # Preferred retinal azimuth of every cell: peak of its GAIN-model pillar field (eye at its mean) over the
    # 10 deg retinal bins with >= 2 s of light, as notebook Q1 does for tuned cells.
    m = use & light & fin
    b = np.floor(((rho[m] + 180.0) % 360.0) / 10.0).astype(int) % len(Q_GRID)
    samp = np.bincount(b, minlength=len(Q_GRID)) * dt >= 2.0
    field = _vm(Q_GRID) @ (coef / x_sd[:, None])[n_nuis + 1:n_nuis + 1 + N_VM]
    d['pref_azimuth'] = Q_GRID[samp][np.argmax(field[samp], 0)] if samp.sum() >= 3 else np.full(C, np.nan)
    return d


X_DPRIME, X_LOG_BF, X_DRAWS = 2.0, np.log(10.0), 200


def anchoring_calls():
    """ Notebook X (supp. X1): per-cell eye- vs head-anchored pillar field, with eye-position gain in both.

    Each tuned cell was simulated 200 times as eye-anchored and as head-anchored (own fit + own shifted
    residual). d' = separation of the two simulated score distributions; the real score's log Bayes factor
    between them gives the call ('eye' / 'head' at 10:1, only where d' >= 2; 'neither' when the real score
    is outside both). Simulated cells are called in the wrong frame 1-2% of the time (X2).
    """
    import pandas as pd
    rows = []
    for sess, r in store_step('X')['XS'].items():
        if 'pillar_cells' not in r:
            continue
        E, H, d = r['pillar_sim_eye'][:X_DRAWS], r['pillar_sim_head'][:X_DRAWS], r['pillar_d_real']
        mE, sE, mH, sH = E.mean(0), np.maximum(E.std(0), 1e-12), H.mean(0), np.maximum(H.std(0), 1e-12)
        dprime = (mH - mE) / np.sqrt(np.maximum((sE ** 2 + sH ** 2) / 2, 1e-30))
        log_bf = (-0.5 * ((d - mE) / sE) ** 2 - np.log(sE)) - (-0.5 * ((d - mH) / sH) ** 2 - np.log(sH))
        pct = lambda S: np.minimum((1 + 2 * np.minimum((S >= d).sum(0), (S <= d).sum(0))) / (1 + len(S)), 1.0)
        call = np.where(log_bf > X_LOG_BF, 'eye', np.where(log_bf < -X_LOG_BF, 'head', 'undecided')).astype(object)
        call[(pct(E) < 0.01) & (pct(H) < 0.01)] = 'neither'
        call[~(dprime >= X_DPRIME)] = 'unidentifiable'
        rows.append(pd.DataFrame({'session': sess, 'cell': np.asarray(r['pillar_cells'], int),
                                  'anchor_dprime': dprime, 'anchor_call': call}))
    return pd.concat(rows, ignore_index=True)


def _build_cell_table():
    import pandas as pd
    rows = [pd.DataFrame(_session_cells(*s)) for s in neural_sessions()]
    T = pd.concat(rows, ignore_index=True)
    enc = store_step('O')['ENC_LD']
    for k in ('dark_noeye_sig', 'light_noeye_matched_sig'):
        T[k] = np.nan
        for sess, idx in T.groupby('session').groups.items():
            if sess in enc and k in enc[sess]:
                T.loc[idx, k] = np.asarray(enc[sess][k], float)[T.loc[idx, 'cell'].to_numpy()]
    Q = store_step('Q')
    QC = Q['QC'].reset_index(drop=True)
    QC['q_row'] = np.arange(len(QC))
    T = T.merge(QC[['session', 'cell', 'q_row', 'gain_cell', 'k']], on=['session', 'cell'], how='left')
    T['q_row'] = T['q_row'].fillna(-1).astype(int)
    T['gain_cell'] = T['gain_cell'].astype(float)
    fsi = field_shift_cells()
    F = pd.concat([pd.DataFrame({'session': fsi[a]['session'], 'cell': fsi[a]['cell'].astype(int),
                                 'field_shift': fsi[a]['index']}) for a in AREAS], ignore_index=True)
    T = T.merge(F, on=['session', 'cell'], how='left')
    T = T.merge(anchoring_calls(), on=['session', 'cell'], how='left')
    T['anchor_call'] = T['anchor_call'].fillna('not tested')
    with np.errstate(divide='ignore', invalid='ignore'):
        T['log2_dark_light'] = np.log2(T['rate_dark'] / T['rate_light'])
    return T, {'qgp_real': Q['QGP']['real'], 'qgp_null': Q['QGP']['null'], 'qvis': np.asarray(Q['QVIS'], bool)}


def cell_table():
    """ Every cell of every animal (one recording per position), with what Figures 9-11 need. Cached. """
    import pandas as pd
    if os.path.isfile(CELL_CACHE):
        with h5py.File(CELL_CACHE, 'r') as f:
            T = pd.DataFrame({k: (v[()].astype(str) if v.dtype.kind == 'S' else v[()]) for k, v in f['cells'].items()})
            return T, {k: f[k][()] for k in ('qgp_real', 'qgp_null', 'qvis')}
    T, Q = _build_cell_table()
    os.makedirs(os.path.dirname(CELL_CACHE), exist_ok=True)
    with h5py.File(CELL_CACHE, 'w') as f:
        for k in T.columns:
            v = T[k].to_numpy()
            f['cells/' + k] = v.astype('S') if v.dtype == object else v
        for k, v in Q.items():
            f[k] = v
    print('saved', CELL_CACHE)
    return T, Q


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    c = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return p, c - h, c + h


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate(np.minimum(1.0, p[o] * (len(p) - np.arange(len(p)))))
    out = np.empty_like(adj)
    out[o] = adj
    return out


def dark_panels(axF, axG):
    """ Fig 1F-G (was 9A-B): do the cells called head-centered in the light keep responding / encoding in the dark?

    Cells grouped by Figure 1E's call in the light (eye, head, or not pillar-tuned). F: each cell's dark / light
    event rate (median, bootstrap 95 % interval). G: fraction of cells whose eye-free encoding model beats its
    null, light (light data cut to the dark's amount) vs dark (Wilson 95 %). Brackets: head vs eye, Mann-Whitney
    (F) and Fisher on the dark bars (G), Holm-corrected over these two tests.
    """
    from scipy.stats import fisher_exact, mannwhitneyu
    T, _ = cell_table()
    T = T[T['area'].isin(AREAS)].reset_index(drop=True)
    grp = np.where(T['eye_centred'], 'eye', np.where(T['head_centred'], 'head',
                                                     np.where(T['tuned'].to_numpy(bool), '', 'untuned')))
    groups = (('eye', EYE, EYE_DARK), ('head', HEAD, HEAD_DARK), ('untuned', SHUF, SHUF_DARK))
    rng = np.random.default_rng(9)

    def frac(col, m):
        v = T.loc[m, col].to_numpy(float)
        v = v[np.isfinite(v)]
        return wilson(v.sum(), len(v)) + (int(v.sum()), len(v))

    ratio = T['rate_dark'].to_numpy() / np.where(T['rate_light'].to_numpy() > 0, T['rate_light'].to_numpy(), np.nan)
    ok = np.isfinite(ratio)
    a, b = frac('dark_noeye_sig', grp == 'head'), frac('dark_noeye_sig', grp == 'eye')
    p = holm([mannwhitneyu(ratio[ok & (grp == 'head')], ratio[ok & (grp == 'eye')]).pvalue,
              fisher_exact([[a[3], a[4] - a[3]], [b[3], b[4] - b[3]]])[1]])

    hi = 0
    for i, (g, _, dark) in enumerate(groups):
        v = ratio[ok & (grp == g)]
        boot = np.median(rng.choice(v, (1000, len(v))), 1)
        axF.bar(i * SP, np.median(v), width=0.34, color=dark)
        errbar(axF, i * SP, *np.percentile(boot, [2.5, 97.5]))
        hi = max(hi, np.percentile(boot, 97.5))
    bracket(axF, 0, SP, 1.06, p[0])
    axF.set_ylim(0, 1.05 * hi)
    axF.set_ylabel('dark / light rate')

    hi = 0
    for i, (g, light, dark) in enumerate(groups):
        for j, (col, color) in enumerate((('light_noeye_matched_sig', light), ('dark_noeye_sig', dark))):
            y, lo, h, _, _ = frac(col, grp == g)
            x = i * SP + (j - 0.5) * 0.38
            axG.bar(x, y, width=0.34, color=color)
            errbar(axG, x, lo, h)
            hi = max(hi, h)
    axG.axhline(0.05, color=INK2, lw=0.57, ls='--', zorder=0)
    bracket(axG, 0.19, SP + 0.19, 1.06, p[1])
    axG.set_ylim(0, 1.05 * hi)
    axG.set_ylabel('frac. cells with\nsig. encoding')
    legend_above(axG, [Patch(color='#9a9a9a'), Patch(color='#3a3a3a')], ['light', 'dark'], 2, y=1.16)
    for ax in (axF, axG):
        ax.set_xticks(np.arange(3) * SP)
        ax.set_xticklabels(['eye', 'head', 'not\ntuned'])
        ax.set_xlabel('call in light')
    for g, *_ in groups:
        m = grp == g
        print('  {}: n={}, dark/light rate {:.3f}, sig. dark {:.3f}, sig. light (matched) {:.3f}'.format(
            g, int(m.sum()), np.median(ratio[ok & m]), frac('dark_noeye_sig', m)[0],
            frac('light_noeye_matched_sig', m)[0]))
    print('  Holm p (rate, dark encoding):', np.round(p, 3))


def load_atlas():
    import json
    with open(ATLAS) as f:
        raw = json.load(f)
    return {k: np.asarray(v, float) for k, v in raw.items() if v is not None and len(v) >= 3}


def map_cells(T, atlas):
    """ Cells for the maps: 'border' cells (< 5 px from another area) go to the area they sit in, or the nearest one. """
    T = T.copy()
    b = np.flatnonzero(T['area'].to_numpy() == 'border')
    xy = T[['x', 'y']].to_numpy(float)[b]
    names = list(atlas)
    d = np.stack([np.min(np.hypot(xy[:, None, 0] - atlas[k][None, :, 0], xy[:, None, 1] - atlas[k][None, :, 1]), 1)
                  for k in names], 1)
    for j, k in enumerate(names):
        d[Path(atlas[k]).contains_points(xy), j] = -1.0
    T.loc[T.index[b], 'area'] = np.array(names)[np.argmin(d, 1)]
    return T[T['area'].isin(AREAS)].reset_index(drop=True)


def map_grid(atlas, step):
    xy = np.concatenate([atlas[a] for a in AREAS])
    (x0, y0), (x1, y1) = np.floor(xy.min(0)) - 4, np.ceil(xy.max(0)) + 4
    gx, gy = np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step)
    X, Y = np.meshgrid(gx, gy)
    ext = (gx[0] - step / 2, gx[-1] + step / 2, gy[-1] + step / 2, gy[0] - step / 2)
    return np.c_[X.ravel(), Y.ravel()], X.shape, ext


def _d2(pts, cxy):
    return ((pts[:, None, :] - cxy[None, :, :]) ** 2).sum(-1)


def _adaptive(pts, cxy, k, smin, smax):
    """ Gaussian weights whose SD grows from smin until k cells are within it; ok = reached k by smax. """
    d2 = _d2(pts, cxy)
    kk = min(k, len(cxy)) - 1
    dk = np.sqrt(np.partition(d2, kk, axis=1)[:, kk])
    s = np.clip(dk, smin, smax)
    return np.exp(-0.5 * d2 / s[:, None] ** 2), (dk <= smax) & (len(cxy) >= k)


def footprint(T, atlas, pts, chunk=1000):
    """ Pixels inside an area's outline and near that area's imaged cells (any cell): the same for every panel. """
    foot = np.zeros(len(pts), bool)
    for a in AREAS:
        inside = np.flatnonzero(Path(atlas[a]).contains_points(pts))
        cxy = T.loc[T['area'].to_numpy() == a, ['x', 'y']].to_numpy(float)
        for i in range(0, len(inside), chunk):
            ix = inside[i:i + chunk]
            foot[ix] = np.exp(-0.5 * _d2(pts[ix], cxy) / MAP_SIGMA ** 2).sum(1) >= FOOT_MIN_W
    return foot


def local_gain_maps(T, Q, atlas, cols, chunk=1000):
    """ Notebook Q4's gain signal, coverage and effective dimensions, computed for the cells around each point.

    Q4 averages every tuned cell's gain-profile outer product (real minus
    shuffled-eye) over a whole area. Here the average is weighted by a
    Gaussian of distance, within one area, so each map pixel is the Q4 value
    of the local population (SD from COV_SIGMA, widened up to COV_SIGMA_MAX
    until COV_K cells are inside). tr(C^2) is split-half (COV_SPLITS random
    halves of the area's cells), as in Q4. Coverage and dimensions divide by
    the local gain signal, so they are shown only where that signal is > 2 SE
    above zero (SE of tr C from the split halves) and the dimension count is
    possible (<= basis size); elsewhere in the footprint they are grey.
    """

    U, P = gain_bases(cols)
    pts, shape, ext = map_grid(atlas, MAP_STEP)
    foot = footprint(T, atlas, pts)
    out = {k: np.full(len(pts), np.nan) for k in ('need', 'coverage', 'eff_dims')}
    rng = np.random.default_rng(10)
    rows = T['q_row'].to_numpy()
    diag = np.arange(U.shape[1]) * (U.shape[1] + 1)
    for a in AREAS:
        inside = np.flatnonzero(Path(atlas[a]).contains_points(pts) & foot)
        sel = np.flatnonzero((T['area'].to_numpy() == a) & (rows >= 0))
        R, N = Q['qgp_real'][rows[sel]][:, cols] @ U, Q['qgp_null'][rows[sel]][:, cols] @ U
        ok = np.all(np.isfinite(R), 1) & np.all(np.isfinite(N), 1)
        sel, R, N = sel[ok], R[ok], N[ok]
        if len(sel) < 10 or not len(inside):
            continue
        M = (R[:, :, None] * R[:, None, :] - N[:, :, None] * N[:, None, :]).reshape(len(sel), -1)
        need = (((R @ P) ** 2).sum(1) - ((N @ P) ** 2).sum(1)) / np.trace(P.T @ P)
        trace = (R ** 2).sum(1) - (N ** 2).sum(1)
        cxy = T.loc[sel, ['x', 'y']].to_numpy(float)
        halves = [rng.permutation(len(sel)) < len(sel) // 2 for _ in range(COV_SPLITS)]
        for i in range(0, len(inside), chunk):
            ix = inside[i:i + chunk]
            W, reach = _adaptive(pts[ix], cxy, COV_K, COV_SIGMA, COV_SIGMA_MAX)
            wsum = W.sum(1)
            nd, tc = W @ need / wsum, W @ trace / wsum
            c2, dtc = np.zeros(len(ix)), np.zeros(len(ix))
            for h in halves:
                CA = W[:, h] @ M[h] / W[:, h].sum(1, keepdims=True)
                CB = W[:, ~h] @ M[~h] / W[:, ~h].sum(1, keepdims=True)
                c2 += (CA * CB).sum(1) / COV_SPLITS
                dtc += (CA[:, diag].sum(1) - CB[:, diag].sum(1)) ** 2 / COV_SPLITS
            good = reach & (tc > np.sqrt(dtc)) & (c2 > 0) & (tc ** 2 <= U.shape[1] * c2)
            out['need'][ix] = np.where(reach, 100 * nd, np.nan)
            out['coverage'][ix] = np.where(good, nd * tc / c2, np.nan)
            out['eff_dims'][ix] = np.where(good, tc ** 2 / c2, np.nan)
    return {k: (v.reshape(shape), (foot & ~np.isfinite(v)).reshape(shape)) for k, v in out.items()}, ext


def two_pole(lo, hi, mid='#f2f1ed'):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('two_pole', [lo, mid, hi])


def compass(ax):
    kw = dict(xycoords='axes fraction', textcoords='axes fraction', fontsize=FONT, color=INK2,
              arrowprops=dict(arrowstyle='<-', color=INK2, lw=0.8))
    ax.annotate('anterior', xy=(0.04, 0.06), xytext=(0.04, 0.24), ha='center', va='bottom', **kw)
    ax.annotate('medial', xy=(0.04, 0.06), xytext=(0.24, 0.06), ha='left', va='center', **kw)



# ---------------------------------------------------------------------------
# Cortex maps (Figure 5): one dot per cell, coloured by its own value.
# ---------------------------------------------------------------------------

SCATTER_S = 1.2                # dot area, pt^2
SCATTER_MID = '#a9a8a3'        # middle of the two-pole colours: the maps' near-white would vanish on white paper


def blues_from(lo=0.2):
    """ 'Blues' without its near-white end, so the lowest values stay visible as dots. """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('blues_from', plt.get_cmap('Blues')(np.linspace(lo, 1, 256)))


def sample_map(M, ext, xy, step=MAP_STEP):
    """ Map value at each cell's position (nearest map pixel; NaN where the map has none). """
    j = np.clip(np.round((xy[:, 0] - ext[0]) / step - 0.5).astype(int), 0, M.shape[1] - 1)
    i = np.clip(np.round((xy[:, 1] - ext[3]) / step - 0.5).astype(int), 0, M.shape[0] - 1)
    return M[i, j]


def cell_gain_signal(T, Q, cols):
    """ Each cell's own term of the gain-signal map (x100): its gain profile's power along the needed shapes,
    real minus shuffled eye (local_gain_maps' 'need' before the spatial average). """
    U, P = gain_bases(cols)
    out = np.full(len(T), np.nan)
    rows = T['q_row'].to_numpy()
    sel = np.flatnonzero(rows >= 0)
    R, N = Q['qgp_real'][rows[sel]][:, cols] @ U, Q['qgp_null'][rows[sel]][:, cols] @ U
    out[sel] = 100 * (((R @ P) ** 2).sum(1) - ((N @ P) ** 2).sum(1)) / np.trace(P.T @ P)
    return out


def cell_scatter(fig, ax, T, v, atlas, cmap=None, label='', lims=None, ticks=None, cats=None, names=False,
                 extend='neither', missing='other cells', seed=0):
    """ One dot per cell at its atlas position, coloured by its own value.

    Cells without a value (NaN) are drawn grey underneath; the rest are drawn in a
    random order so no group always sits on top. cats = [(value, colour, label)]
    gives a legend above the panel in place of a colour bar.
    """
    from matplotlib.colors import Normalize, to_hex
    xy = T[['x', 'y']].to_numpy(float)
    v = np.asarray(v, float)
    has = ~np.isnan(v)
    kw = dict(s=SCATTER_S, lw=0, rasterized=True)
    ax.scatter(*xy[~has].T, c=NO_DATA, **kw)
    o = np.random.default_rng(seed).permutation(np.flatnonzero(has))
    handles, labels = [], []
    if cats:
        col = np.full(len(v), None, object)
        for val, c, lab in cats:
            c = to_hex(c)
            col[v == val] = c
            handles.append(Patch(color=c))
            labels.append(lab)
        sc = ax.scatter(*xy[o].T, c=list(col[o]), **kw)
    else:
        sc = ax.scatter(*xy[o].T, c=np.clip(v[o], *lims), cmap=cmap, norm=Normalize(*lims), **kw)
    cb = fig.colorbar(sc if not cats else plt.cm.ScalarMappable(cmap='Greys'), ax=ax, shrink=0.75, aspect=14,
                      pad=0.02, ticks=ticks, extend=extend)
    if cats:
        cb.ax.set_visible(False)       # keeps every panel the same size as its colour-bar neighbours
    else:
        cb.set_label(label, fontsize=FONT)
        cb.ax.tick_params(labelsize=FONT, width=0.45, length=1.7)
        cb.outline.set_linewidth(0.45)
    for k, c in atlas.items():
        ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=INK2 if k in AREAS else '#b9b8b2', lw=0.6)
        if names and k in AREAS:
            ax.text(*c.mean(0), k, ha='center', va='center', fontsize=FONT, color=INK, fontweight='bold')
    ext = map_grid(atlas, MAP_STEP)[2]
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect('equal')
    ax.axis('off')
    if (~has).any():
        handles.append(Patch(color=NO_DATA))
        labels.append(missing)
    if handles:
        legend_above(ax, handles, labels, 2, y=1.0)


def _lims(v, sym):
    v = v[np.isfinite(v)]
    return (-np.percentile(np.abs(v), 98), np.percentile(np.abs(v), 98)) if sym else tuple(np.percentile(v, [2, 98]))


def figure5(D):
    """ Where on the cortex: one dot per cell at its atlas position. Old Figure10_cells A, D, G, H, J, K.

    Left column: which cells are pillar-tuned (A) and gain cells (D). Middle and right: visible-field (top) and
    never-visible-field (bottom) gain signal per cell (B, E) and the coverage of the local population around each
    cell with a gain profile (C, F; the population map sampled at the cell). Continuous colours are clipped at the
    cells' 2nd / 98th percentile.
    """
    T, Q = cell_table()
    atlas = load_atlas()
    T = map_cells(T, atlas)
    bl = plt.get_cmap('Blues')
    lo_b, hi_b = bl(0.3), bl(0.85)
    xy = T[['x', 'y']].to_numpy(float)
    profiled = (T['q_row'] >= 0).to_numpy()
    fig = plt.figure(figsize=(WIDTH, 6.0), layout='constrained')
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    cell_scatter(fig, ax, T, T['tuned'].to_numpy(float), atlas, cats=[(0, lo_b, 'not tuned'), (1, hi_b, 'pillar-tuned')],
                 names=True)
    compass(ax)
    letter(ax, 'A', x=0, y=0)
    ax = fig.add_subplot(gs[1, 0])
    gain = np.where(T['q_row'] >= 0, T['gain_cell'].to_numpy(float), np.nan)
    cell_scatter(fig, ax, T, gain, atlas, cats=[(0, lo_b, 'no gain'), (1, hi_b, 'gain cell')])
    letter(ax, 'D', x=0, y=0)

    for r, (cols, where) in enumerate(((Q['qvis'], 'visible'), (~Q['qvis'], 'never visible'))):
        G, ext = local_gain_maps(T, Q, atlas, cols)
        need = cell_gain_signal(T, Q, cols)
        ax = fig.add_subplot(gs[r, 1])
        cell_scatter(fig, ax, T, need, atlas, blues_from(), '{}:\ngain signal (x100), per cell'.format(where),
                     lims=_lims(need, False), extend='both')
        letter(ax, 'BE'[r], x=0, y=0)
        ax = fig.add_subplot(gs[r, 2])
        v = np.where(profiled, sample_map(G['coverage'][0], ext, xy), np.nan)
        cell_scatter(fig, ax, T, v, atlas, blues_from(), '{}: coverage\n(local population)'.format(where),
                     lims=_lims(G['coverage'][0], False), extend='both', missing='no value here / no profile')
        letter(ax, 'CF'[r], x=0, y=0)
    save(fig, 'Figure5')


# ---- Eye-in-head vs gaze-in-world gain (notebook section AE, store step 'AE') ----
GAZE = '#8c564b'               # gaze-in-world (new frame colour; eye-in-head simulations use EYE_GAIN as in Fig 2C-D)
GAZE_SRCS = (('real', REAL, 'real'), ('sim_eye', EYE_GAIN, 'simulated eye-in-head'),
             ('sim_gw', GAZE, 'simulated gaze-in-world'))
GAZE_DRAWS, GAZE_SHUFFLES, GAZE_BOOT = 2, 10, 1000
F_GAZE_TOP = [2.05, 0.6, 2.05, 1.0]   # A, gap (A's side labels), B, spacer
F_GAZE_BOTTOM = [2.45, 2.65, 0.6]  # C, D, spacer


def gaze_vals(d, k, src):
    """ Fraction of cells with call k (opp_cos: median): real, or pooled over the simulation draws / shuffles. """
    if src == 'real':
        v = d[k].to_numpy(float)
    else:
        n = GAZE_SHUFFLES if src == 'null' else GAZE_DRAWS
        v = np.concatenate([d['{}_{}_{}'.format(src, k, j)].to_numpy(float) for j in range(n)])
    return np.nanmedian(v) if k == 'opp_cos' else np.nanmean(v)


def gaze_pi(d):
    """ Fraction behaving as gaze-in-world: the mix of the two simulations whose score (eye_better - gw_better)
    matches the real cells'. """
    S = {s: gaze_vals(d, 'eye_better', s) - gaze_vals(d, 'gw_better', s) for s, _, _ in GAZE_SRCS}
    return (S['sim_eye'] - S['real']) / (S['sim_eye'] - S['sim_gw'])


def gaze_figure(D):
    """ Does the eye-position gain follow eye-in-head or gaze-in-world (heading + eye) position? All J-tuned cells.

    A: head-to-head call (up: eye-in-head model better; down: gaze-in-world better), real cells against each cell
    simulated from its own eye-in-head or gaze-in-world fit. B: the eye gain changes with heading (an e x heading
    gain adds beyond the eye-in-head model). C: estimated fraction of gaze-in-world cells (0 = eye-in-head), 95%
    bootstrap over cells. D: median cosine between the eye gain profiles fit in opposite heading quadrants
    (eye-in-head +, gaze-in-world flips sign).
    """
    T = store_step('AE')['AEC']
    T = T[T['area'].isin(AREAS)]               # no PM, so 'all' pools V1, RL, A and AM
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(WIDTH, 5.6), layout='constrained')
    gs = fig.add_gridspec(2, 1, height_ratios=[2.8, 2.8])
    top = gs[0].subgridspec(1, 4, width_ratios=F_GAZE_TOP)
    bottom = gs[1].subgridspec(1, 3, width_ratios=F_GAZE_BOTTOM)
    w = 0.26

    ax = fig.add_subplot(top[0])
    lo = hi = 0
    for i, a in enumerate(AREAS):
        d = T[T['area'] == a]
        for j, (src, color, _) in enumerate(GAZE_SRCS):
            x, sim = i + (j - 1) * w, src != 'real'
            kw = dict(width=w * 0.9, color=color, hatch=HATCH if sim else None, edgecolor=INK2 if sim else 'none', lw=0.28)
            up, dn = gaze_vals(d, 'eye_better', src), gaze_vals(d, 'gw_better', src)
            ax.bar(x, up, **kw)
            ax.bar(x, -dn, **kw)
            hi, lo = max(hi, up), min(lo, -dn)
    ax.axhline(0, color=INK2, lw=0.45)
    area_ticks(ax)
    ax.set_ylim(1.1 * lo, 1.1 * hi)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: '{:g}'.format(abs(round(v, 2)))))
    ax.set_ylabel('frac. tuned cells')
    ax.text(1.02, 0.75, 'eye-in-head\nbetter', transform=ax.transAxes, color=EYE_GAIN, va='center', fontsize=FONT)
    ax.text(1.02, 0.2, 'gaze-in-world\nbetter', transform=ax.transAxes, color=GAZE, va='center', fontsize=FONT)
    legend_above(ax, [Patch(facecolor=c, hatch=HATCH if s != 'real' else None, edgecolor=INK2 if s != 'real' else 'none')
                      for s, c, _ in GAZE_SRCS], [l for _, _, l in GAZE_SRCS], 2)
    letter(ax, 'A')

    ax = fig.add_subplot(top[2])
    hi = 0
    for i, a in enumerate(AREAS):
        d = T[T['area'] == a]
        for j, (src, color, _) in enumerate(GAZE_SRCS):
            x, sim = i + (j - 1) * w, src != 'real'
            v = gaze_vals(d, 'eye_varies', src)
            ax.bar(x, v, width=w * 0.9, color=color, hatch=HATCH if sim else None, edgecolor=INK2 if sim else 'none',
                   lw=0.28)
            if not sim:
                k = int(d['eye_varies'].sum())
                _, l, h = wilson(k, len(d))
                errbar(ax, x, l, h)
                v = h
            hi = max(hi, v)
    area_ticks(ax)
    ax.set_ylim(0, 1.05 * hi)
    ax.set_ylabel('frac. tuned cells,\neye gain varies with heading')
    letter(ax, 'B')

    ax = fig.add_subplot(bottom[0])
    ax.axhline(0, color=EYE_GAIN, lw=0.85, ls='--')
    ax.axhline(1, color=GAZE, lw=0.85, ls='--')
    names = AREAS + ['all']
    for i, a in enumerate(names):
        d = T if a == 'all' else T[T['area'] == a]
        b = [gaze_pi(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(GAZE_BOOT)]
        color = AREA_COLOR.get(a, INK)
        ax.plot([i, i], np.nanpercentile(b, [2.5, 97.5]), color=color, lw=ERR_LW, zorder=3)
        ax.plot(i, gaze_pi(d), 'o', color=color, ms=4.5, zorder=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel('est. frac. gaze-in-world')
    ax.text(1.02, 0, 'all eye-\nin-head', transform=ax.get_yaxis_transform(), color=EYE_GAIN, va='center', fontsize=FONT)
    ax.text(1.02, 1, 'all gaze-\nin-world', transform=ax.get_yaxis_transform(), color=GAZE, va='center', fontsize=FONT)
    letter(ax, 'C')

    ax = fig.add_subplot(bottom[1])
    srcs = GAZE_SRCS + (('null', SHUF, 'eye shuffled'),)
    w4 = 0.2
    lo = hi = 0
    for i, a in enumerate(AREAS):
        d = T[T['area'] == a]
        for j, (src, color, _) in enumerate(srcs):
            sim = src.startswith('sim')
            v = gaze_vals(d, 'opp_cos', src)
            ax.bar(i + (j - 1.5) * w4, v, width=w4 * 0.9, color=color, hatch=HATCH if sim else None,
                   edgecolor=INK2 if sim else 'none', lw=0.28)
            lo, hi = min(lo, v), max(hi, v)
    ax.axhline(0, color=INK2, lw=0.45)
    area_ticks(ax)
    ax.set_ylim(min(0, 1.2 * lo), 1.1 * hi)
    ax.set_ylabel('eye gain, opposite\nheadings (median cos)')
    legend_above(ax, [Patch(color=SHUF)], ['eye shuffled'], 1)
    letter(ax, 'D')
    save(fig, 'Gaze_figure')


FIGURES = {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5, 'ebc1': figure6, 'ebc2': figure7,
           'ebc3': figure8, 'gaze': gaze_figure}

if __name__ == '__main__':
    data = load()
    which = [int(a) if a.isdigit() else a for a in sys.argv[1:]] or list(FIGURES)
    for n in which:
        FIGURES[n](data)
