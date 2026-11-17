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
OUT = os.path.join(HERE, 'manuscript_figures_py')
FORMATS = ('png', 'pdf')
DPI = 300

WIDTH = 8.5
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


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for fmt in FORMATS:
        fig.savefig(os.path.join(OUT, '{}.{}'.format(name, fmt)), dpi=DPI if fmt == 'png' else None)
    plt.close(fig)
    print('saved', os.path.join(OUT, name))


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


def figure1(D):
    d = D['fig1']
    fig = plt.figure(figsize=(WIDTH, 10.75), layout='constrained')
    gs = fig.add_gridspec(4, 1, height_ratios=[3.29, 2.05, 2.05, 3.36])
    top = gs[0].subgridspec(1, 3, width_ratios=[2.1, 1.25, 0.55])
    row_d = gs[1].subgridspec(1, 4)
    row_e = gs[2].subgridspec(1, 4)
    bottom = gs[3].subgridspec(1, 2)

    ax = fig.add_subplot(top[0])
    ax.set_box_aspect(3 / 4)
    rng = np.random.default_rng(1)
    w = 0.4
    for i, a in enumerate(AREAS):
        for j, (col, dark) in enumerate((('gm_tuned', False), ('gm_tuned_dark', True))):
            v = d['A'][a][col]
            x = i + (j - 0.5) * w
            ax.bar(x, v['frac'], width=w * 0.9, color=AREA_DARK[a] if dark else AREA_COLOR[a])
            s = v['sessions']
            ax.scatter(x + rng.uniform(-0.09, 0.09, len(s)), s, s=3.85, color=INK, lw=0, zorder=3)
            errbar(ax, x, v['lo'], v['hi'])
    area_ticks(ax)
    ax.set_ylabel('frac. cells')
    ax.set_ylim(0, 1.05 * max(d['A'][a][c]['hi'] for a in AREAS for c in ('gm_tuned', 'gm_tuned_dark')))
    legA = legend_above(ax, [Patch(color='#9a9a9a'), Patch(color='#3a3a3a')], ['light', 'dark'], 2)
    letter(ax, 'A')

    ax = axB = fig.add_subplot(top[1])
    ax.set_box_aspect(3 / 4)
    ax.set_anchor('C')
    e = d['B']['bin_edges']
    ax.bar(e[:-1] + np.diff(e) / 2, d['B']['frac_frames'], width=np.diff(e) * 0.9, color='#8a8986')
    for v in d['B']['third_cuts']:
        ax.axvline(v, color=INK, lw=1.36, ls='--')
    ax.set_xlim(e[0], e[-1])
    ax.set_xlabel('eye in head (deg)')
    ax.set_ylabel('frac. of frames')
    letter(ax, 'B', level=legA)

    ax = axC = fig.add_subplot(top[2])
    ax.set_anchor('C')
    g = d['C']['gap_deg']
    ax.scatter(np.random.default_rng(4).uniform(-0.15, 0.15, len(g)), g, s=6.42, color=INK2, lw=0, zorder=3)
    ax.plot([-0.3, 0.3], [np.median(g)] * 2, color=INK, lw=1.42)
    ax.axhline(d['C']['min_gap'], color=INK2, lw=0.57, ls='--')
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylim(bottom=0)
    ax.set_ylabel('gap (deg)')
    letter(ax, 'C', x=-34, level=legA)

    for j, a in enumerate(AREAS):
        c = d['DE'][a]
        grid, lo, hi, dl = c['grid'], c['low'], c['high'], c['delta']
        ax = fig.add_subplot(row_d[j])
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
            letter(ax, 'D')
        ax = fig.add_subplot(row_e[j])
        ax.set_box_aspect(3 / 4)
        for x, y, ls in (((grid - dl / 2 + 180) % 360 - 180, lo, LS_LOW), ((grid + dl / 2 + 180) % 360 - 180, hi, LS_HIGH)):
            o = np.argsort(x)
            ax.plot(x[o], y[o], color=AREA_COLOR[a], lw=1.59, ls=ls)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel('head-centered\nazimuth (deg)')
        if j == 0:
            ax.set_ylabel('rate')
            letter(ax, 'E')

    ax = fig.add_subplot(bottom[0])
    ax.set_box_aspect(3 / 4)
    fsi = field_shift_cells({a: d['F'][a]['ratio'] for a in AREAS})
    rng = np.random.default_rng(3)
    half = 0.36
    for i, a in enumerate(AREAS):
        y = fsi[a]['index']
        y = y[np.isfinite(y)]
        vp = ax.violinplot([y], positions=[i], widths=2 * half, showextrema=False, showmedians=False)
        for b in vp['bodies']:
            b.set_facecolor(AREA_COLOR[a])
            b.set_edgecolor(INK)
            b.set_linewidth(0.79)
            b.set_alpha(0.55)
        kde = gaussian_kde(y)
        dens = kde(y) / kde(np.linspace(y.min(), y.max(), 200)).max()
        ax.scatter(i + rng.uniform(-1, 1, len(y)) * half * dens * 0.9, y, s=1.6, color=INK, lw=0, alpha=0.6,
                   zorder=3)
        ax.plot([i - half * 0.6, i + half * 0.6], [np.median(y)] * 2, color=INK, lw=1.59, zorder=4)
    ax.axhline(0, color=INK2, lw=0.57, ls='--', zorder=0)
    area_ticks(ax)
    ax.set_xlim(-0.6, len(AREAS) - 0.4)
    ax.set_ylabel('field-shift index')
    letter(ax, 'F')

    ax = fig.add_subplot(bottom[1])
    ax.set_box_aspect(3 / 4)
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
    legend_above(ax, [Patch(color=c) for _, c in cols],
                 ['eye (light)', 'eye (dark)', 'head (light)', 'head (dark)'], 2)
    letter(ax, 'G')
    fig.canvas.draw()
    W, H = fig.get_size_inches()
    axC.set_box_aspect(axB.get_position().height * H / (axC.get_position().width * W))
    save(fig, 'Figure1')


def figure2(D):
    d = D['fig2']
    fig = plt.figure(figsize=(WIDTH, 8.64), layout='constrained')
    gs = fig.add_gridspec(3, 1, height_ratios=[2.07, 4.02, 2.55])
    row_a = gs[0].subgridspec(1, 4)
    mid = gs[1].subgridspec(1, 2)
    row_c = mid[1].subgridspec(2, 2)
    row_d = gs[2].subgridspec(1, 4, wspace=0.12)

    for j, a in enumerate(AREAS):
        c = d['A'][a]
        ax = fig.add_subplot(row_a[j])
        ax.set_title('cell {}'.format(j + 5), fontsize=FONT, pad=4)
        ax.axvspan(*VISIBLE, color=VIS_SHADE, lw=0)
        ax.plot(c['azimuth'], c['eye_low'], color=AREA_COLOR[a], lw=1.59, ls=LS_LOW)
        ax.plot(c['azimuth'], c['eye_mean'], color=INK2, lw=1.02)
        ax.plot(c['azimuth'], c['eye_high'], color=AREA_COLOR[a], lw=1.59, ls=LS_HIGH)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel('retinal azimuth (deg)')
        if j == 0:
            ax.set_ylabel('pillar field')
            legend_above(ax, [Line2D([], [], color=INK, lw=1.59, ls=LS_LOW), Line2D([], [], color=INK2, lw=1.02),
                              Line2D([], [], color=INK, lw=1.59, ls=LS_HIGH)], ['eye -1 SD', 'eye at mean', 'eye +1 SD'], 3, y=1.15)
            letter(ax, 'A')

    ax = fig.add_subplot(mid[0])
    ax.set_box_aspect(3 / 4)
    ax.set_anchor('N')
    for i, a in enumerate(AREAS):
        r, s = d['B'][a]['real'], d['B'][a]['shuffled']
        ax.bar(i - 0.18, r['frac'], width=0.34, color=AREA_COLOR[a])
        errbar(ax, i - 0.18, r['lo'], r['hi'])
        ax.bar(i + 0.18, s['frac'], width=0.34, color=SHUF)
        errbar(ax, i + 0.18, s['lo'], s['hi'])
    area_ticks(ax)
    ax.set_ylim(0, 1.05 * max(max(d['B'][a]['real']['hi'], d['B'][a]['shuffled']['hi']) for a in AREAS))
    ax.set_ylabel('frac. tuned cells')
    legend_above(ax, [Patch(color=INK2), Patch(color=SHUF)], ['real, beyond other modulators', 'eye shuffled in gain term'], 1)
    letter(ax, 'B')

    for j, a in enumerate(AREAS):
        c = d['C'][a]
        ax = fig.add_subplot(row_c[j // 2, j % 2])
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.45, ls='--')
        ax.scatter(c['shift_r'], c['scale_r'], s=12.84, color=AREA_COLOR[a], alpha=0.7, lw=0)
        ax.scatter([np.median(c['shift_r'])], [np.median(c['scale_r'])], s=96.33, marker='+', color=INK, lw=1.42, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('|r| shift' if j >= 2 else '')
        ax.set_ylabel('|r| scale' if j % 2 == 0 else '')
        if j == 0:
            letter(ax, 'C')

    axs = [fig.add_subplot(row_d[j]) for j in range(4)]
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
        for k in np.flatnonzero(c['p'] < 0.05):
            ax.text(cen[k], up[k] + dn[k], '*', ha='center', va='bottom', fontsize=2 * FONT)
        ymax = max(ymax, (up + dn).max())
        ax.set_xlim(e[0], e[-1])
        ax.set_xticks([-90, 0, 90, 180])
        ax.set_xlabel('pref. azimuth (deg)')
        ax.set_box_aspect(3 / 4)
        if j == 0:
            ax.set_ylabel('frac. gain cells')
            legend_above(ax, [Line2D([], [], color=INK, lw=0.57, ls='--'), Patch(color=UP), Patch(color=DOWN)],
                         ['chance', 'gain up with nasal eye', 'gain down with nasal eye'], 3)
            letter(ax, 'D')
        else:
            ax.tick_params(labelleft=False)
    for ax in axs:
        ax.set_ylim(0, 1.2 * ymax)
    save(fig, 'Figure2')


def figure3(D):
    d = D['fig3']
    fig = plt.figure(figsize=(WIDTH, 6.56), layout='constrained')
    gs = fig.add_gridspec(2, 1, height_ratios=[3.58, 2.98])
    top = gs[0].subgridspec(1, 2)
    bottom = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 0.75])

    bars = [('real', REAL), ('sim_eye_gain', EYE_GAIN), ('sim_head', HEAD)]
    labels = ['real', 'simulated eye-centered + gain', 'simulated head-centered']
    axA = fig.add_subplot(top[0])
    axB = fig.add_subplot(top[1], sharey=axA)
    hi = 0
    for ax, panel in ((axA, 'A'), (axB, 'B')):
        w = 0.26
        for i, a in enumerate(AREAS):
            for j, (key, color) in enumerate(bars):
                v = d[panel][a][key]
                x = i + (j - 1) * w
                sim = key != 'real'
                ax.bar(x, v['frac'], width=w * 0.9, color=color, hatch=HATCH if sim else None,
                       edgecolor=INK2 if sim else 'none', lw=0.28)
                errbar(ax, x, v['lo'], v['hi'])
                hi = max(hi, v['hi'])
        area_ticks(ax)
        legend_above(ax, [Patch(facecolor=c, hatch=HATCH if k != 'real' else None, edgecolor=INK2 if k != 'real' else 'none')
                          for k, c in bars], labels, 1)
        letter(ax, panel)
    axA.set_ylim(0, 1.05 * hi)
    axA.set_ylabel('frac. tuned cells')
    axA.set_box_aspect(3 / 4)
    axB.set_box_aspect(3 / 4)
    axB.tick_params(labelleft=True)

    for k, (panel, ylab) in enumerate((('C', 'readout ' + R2), ('D', 'per-cell signal'))):
        ax = fig.add_subplot(bottom[k])
        ax.set_box_aspect(3 / 4)
        hi = 0
        for i, a in enumerate(AREAS):
            for kind, x, color in (('real', i - 0.18, AREA_COLOR[a]), ('null', i + 0.18, SHUF)):
                v = d[panel][a][kind]
                ax.bar(x, v['median'], width=0.34, color=color)
                errbar(ax, x, v['p10'], v['p90'])
                hi = max(hi, v['p90'])
        area_ticks(ax)
        ax.set_ylim(0, 1.05 * hi)
        ax.set_ylabel(ylab)
        if panel == 'D':
            ax.text(0, 1.02, '1e−3', transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT)
        legend_above(ax, [Patch(color=INK2), Patch(color=SHUF)], ['real', 'shuffled eye'], 2, y=1.1)
        letter(ax, panel)

    ax = fig.add_subplot(bottom[2])
    ax.set_box_aspect(1)
    lo, hi = d['E']['lims']
    xx = np.linspace(lo, hi, 2000)
    ax.plot(xx, xx, color=INK2, lw=0.85, ls='--')
    ax.axhline(0, color=GRID, lw=0.34, zorder=0)
    ax.axvline(0, color=GRID, lw=0.34, zorder=0)
    for a in AREAS:
        (nq1, xm, nq3), (rq1, ym, rq3) = d['E'][a]['null'], d['E'][a]['real']
        ax.errorbar(xm, ym, xerr=[[xm - nq1], [nq3 - xm]], yerr=[[ym - rq1], [rq3 - ym]], fmt='o', ms=3.97,
                    color=AREA_COLOR[a], elinewidth=1.81, capsize=0, zorder=3)
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for set_t, set_l in ((ax.set_xticks, ax.set_xticklabels), (ax.set_yticks, ax.set_yticklabels)):
        set_t([-0.1, 0, 0.4])
        set_l(['-0.1', '0', '0.4'])
    ax.minorticks_off()
    ax.set_xlabel('shuffled-eye ' + R2)
    ax.set_ylabel('real ' + R2)
    legend_above(ax, [Line2D([], [], color=AREA_COLOR[a], lw=1.81) for a in AREAS], AREAS, 2, y=1.1)
    letter(ax, 'E')
    save(fig, 'Figure3')


def figure4(D):
    d = D['fig4']
    fig = plt.figure(figsize=(WIDTH, 11.0), layout='constrained')
    gs = fig.add_gridspec(3, 1, height_ratios=[8.6, 6.4, 7.4])
    top = gs[0].subgridspec(2, 4)
    mid = gs[1].subgridspec(1, 2, width_ratios=[2, 2])
    row_e = mid[0].subgridspec(2, 2)
    row_f = mid[1].subgridspec(2, 1)
    row_g = gs[2].subgridspec(2, 4)
    rows = (('visible', 'visible'), ('never_visible', 'never visible'))
    comp_ls = ('-', '--', ':')
    comp_lw = 1.98
    comp_legend = ([Line2D([], [], color=INK, ls=ls, lw=comp_lw) for ls in comp_ls], ['1', '2', '3'])

    def profile_ylim(ax, curves):
        y = np.concatenate([c[np.isfinite(c)] for c in curves])
        ax.set_ylim(min(0.0, 1.1 * y.min()), 1.1 * y.max())

    for r, (key, name) in enumerate(rows):
        ag = d['AG'][key]
        ax = fig.add_subplot(top[r, 0])
        for k in range(3):
            ax.plot(ag['azimuth'], ag['needed'][k], color=INK, ls=comp_ls[k], lw=comp_lw)
        ax.axhline(0, color=INK2, lw=0.34)
        if key == 'never_visible':
            visible_gap(ax, ag['azimuth'], ag['needed'])
        profile_ylim(ax, ag['needed'])
        ax.set_box_aspect(0.8)
        ax.set_xlabel('retinal azimuth\n(deg)')
        ax.set_ylabel('{}:\ngain profile'.format(name))
        if r == 0:
            legend_above(ax, *comp_legend, ncol=3, title='gain component', handlelength=1.6, handletextpad=0.4,
                         columnspacing=0.8, alignment='left')
            letter(ax, 'A')

    for c, (key, ylab) in enumerate((('need', 'gain signal (x100)'), ('coverage', 'coverage'),
                                     ('eff_dims', 'effective gain dimensions'))):
        axs = [fig.add_subplot(top[r, c + 1]) for r in range(2)]
        hi = 0
        for ax, (row, name) in zip(axs, rows):
            for i, a in enumerate(AREAS):
                v = d['BCD'][row][a][key]
                ax.bar(i, v['value'], width=0.6, color=AREA_COLOR[a])
                errbar(ax, i, v['lo'], v['hi'])
                hi = max(hi, v['hi'], v['value'])
            area_ticks(ax, rotate=90)
            ax.set_ylabel('{}:\n{}'.format(name, ylab) if c == 0 else ylab)
        for ax in axs:
            ax.set_ylim(0, 1.1 * hi)
        letter(axs[0], 'BCD'[c], x=-42.5)

    axs = [fig.add_subplot(row_e[j // 2, j % 2]) for j in range(4)]
    for j, (ax, a) in enumerate(zip(axs, AREAS)):
        c = d['E'][a]
        ax.axvspan(*VISIBLE, color=VIS_SHADE, lw=0)
        x = np.r_[c['direction'], c['direction'][0] + 360]
        wrap = lambda y: np.r_[y, y[0]]
        ax.fill_between(x, wrap(c['p10']), wrap(c['p90']), color=AREA_COLOR[a], alpha=0.25, lw=0)
        ax.plot(x, wrap(c['median']), color=AREA_COLOR[a], lw=1.98, marker='o', ms=3.97)
        ax.axhline(0, color=INK2, lw=0.45)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel('head-centered\ndirection (deg)' if j >= 2 else '')
        ax.set_ylabel('real - null ' + R2 if j % 2 == 0 else '')
    lo = min(ax.get_ylim()[0] for ax in axs)
    hi = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(lo, hi)
    letter(axs[0], 'E')

    kap = d['F']['kappas']
    axs = [fig.add_subplot(row_f[r]) for r in range(2)]
    w = 0.8 / len(AREAS)
    hi = 0
    for ax, (row, name) in zip(axs, rows):
        for j, a in enumerate(AREAS):
            v = d['F'][row][a]
            x = np.arange(len(kap)) + (j - (len(AREAS) - 1) / 2) * w
            ax.bar(x, v['median'], width=w * 0.9, color=AREA_COLOR[a])
            for xi, l_, h_ in zip(x, v['p10'], v['p90']):
                errbar(ax, xi, l_, h_)
            hi = max(hi, np.max(v['p90']))
        ax.axhline(0, color=INK2, lw=0.45)
        ax.set_xticks(range(len(kap)))
        ax.set_xticklabels(['{:g}'.format(k) for k in kap])
        ax.set_ylabel('{}:\nreal - null {}'.format(name, R2))
    for ax in axs:
        ax.set_ylim(0, 1.05 * hi)
    axs[1].set_xlabel('target tuning width')
    legend_above(axs[0], [Patch(color=AREA_COLOR[a]) for a in AREAS], AREAS, 4)
    letter(axs[0], 'F')

    for r, (key, name) in enumerate(rows):
        ag = d['AG'][key]
        for j, a in enumerate(AREAS):
            ax = fig.add_subplot(row_g[r, j])
            for k in range(3):
                ax.plot(ag['azimuth'], ag[a][k], color=AREA_COLOR[a], ls=comp_ls[k], lw=comp_lw)
            ax.axhline(0, color=INK2, lw=0.34)
            if key == 'never_visible':
                visible_gap(ax, ag['azimuth'], ag[a])
            profile_ylim(ax, ag[a])
            ax.set_box_aspect(0.8)
            ax.set_xlabel('retinal azimuth\n(deg)')
            if j == 0:
                ax.set_ylabel('{}:\ngain profile'.format(name))
            if r == 0 and j == 0:
                legend_above(ax, *comp_legend, ncol=3, title='gain component', handlelength=1.6, handletextpad=0.4,
                             columnspacing=0.8, alignment='left')
                letter(ax, 'G')
    save(fig, 'Figure4')


def figure5(D):
    d = D['fig5']
    fig = plt.figure(figsize=(WIDTH, 3.4), layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1, 1])

    ax = fig.add_subplot(gs[0])
    ax.set_box_aspect(3 / 4)
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
    legend_above(ax, [Patch(color=HEAD), Patch(color=SHUF), Patch(color=HEAD_DARK), Patch(color=SHUF_DARK)],
                 ['hb gain', 'hb gain, shuffled', 'hb gain beyond turning / speed',
                  'hb gain beyond turning / speed, shuffled'], 1)
    letter(ax, 'A')

    ax = fig.add_subplot(gs[1])
    ax.set_box_aspect(3 / 4)
    rng = np.random.default_rng(3)
    hi = 0
    for i, a in enumerate(AREAS):
        for kind, x, color in (('real', i - 0.18, AREA_COLOR[a]), ('null', i + 0.18, SHUF)):
            v = d['B'][a][kind]
            ax.bar(x, v['median'], width=0.34, color=color)
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v['values'])), v['values'], s=3.85, color=INK, lw=0, zorder=3)
            errbar(ax, x, v['p10'], v['p90'])
            hi = max(hi, np.max(v['values']))
    area_ticks(ax)
    ax.set_ylim(0, 1.05 * hi)
    ax.set_ylabel('per-cell signal')
    ax.text(0, 1.02, '1e-3', transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT)
    legend_above(ax, [Patch(color=INK2), Patch(color=SHUF)], ['real', 'head-on-body shuffled'], 1, y=1.1)
    letter(ax, 'B')

    # C: the same pooled body-specific readout as B, as R2 at a cell count every area has (matched, as Fig 3C).
    ax = fig.add_subplot(gs[2])
    ax.set_box_aspect(3 / 4)
    r2 = body_readout_r2()
    hi = 0
    for i, a in enumerate(AREAS):
        for kind, x, color in (('real', i - 0.18, AREA_COLOR[a]), ('null', i + 0.18, SHUF)):
            v = r2[a][kind]
            ax.bar(x, np.median(v), width=0.34, color=color)
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=3.85, color=INK, lw=0, zorder=3)
            errbar(ax, x, np.percentile(v, 10), np.percentile(v, 90))
            hi = max(hi, np.max(v))
    area_ticks(ax)
    ax.set_ylim(0, 1.05 * hi)
    ax.set_ylabel('readout ' + R2)
    legend_above(ax, [Patch(color=INK2), Patch(color=SHUF)], ['real', 'head-on-body shuffled'], 1, y=1.1)
    letter(ax, 'C')
    save(fig, 'Figure5')


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

    save(fig, 'Figure6')


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

    save(fig, 'Figure7')


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
    save(fig, 'Figure8')


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


def _dark_row(fig, gs, T, grp, xlabel, letters, rng):
    """ One row of Figure 9 for one grouping of the light-period cells ('eye', 'head', 'untuned'). """
    from scipy.stats import fisher_exact, mannwhitneyu
    groups = (('eye', EYE, EYE_DARK), ('head', HEAD, HEAD_DARK), ('untuned', SHUF, SHUF_DARK))
    enough = (T['dark_eye_frames'] >= MIN_DARK_EYE_FRAMES).to_numpy()

    def frac(col, m):
        v = T.loc[m, col].to_numpy(float)
        v = v[np.isfinite(v)]
        return wilson(v.sum(), len(v)) + (int(v.sum()), len(v))

    def fisher(col, m):
        a, b = frac(col, m & (grp == 'head')), frac(col, m & (grp == 'eye'))
        return fisher_exact([[a[3], a[4] - a[3]], [b[3], b[4] - b[3]]])[1]

    ratio = T['rate_dark'].to_numpy() / np.where(T['rate_light'].to_numpy() > 0, T['rate_light'].to_numpy(), np.nan)
    ok_r = np.isfinite(ratio)
    p = holm([mannwhitneyu(ratio[ok_r & (grp == 'head')], ratio[ok_r & (grp == 'eye')]).pvalue,
              fisher('dark_noeye_sig', np.ones(len(T), bool)), fisher('tuned_dark', enough),
              fisher('head_centred_dark', enough)])
    axs = [fig.add_subplot(gs[k]) for k in range(4)]

    ax = axs[0]
    ax.set_box_aspect(3 / 4)
    hi = 0
    for i, (g, _, dark) in enumerate(groups):
        v = ratio[ok_r & (grp == g)]
        boot = np.median(rng.choice(v, (1000, len(v))), 1)
        ax.bar(i, np.median(v), width=0.6, color=dark)
        errbar(ax, i, *np.percentile(boot, [2.5, 97.5]))
        hi = max(hi, np.percentile(boot, 97.5))
    bracket(ax, 0, 1, 1.06, p[0])
    ax.set_ylim(0, 1.05 * hi)
    ax.set_ylabel('dark / light rate')

    ax = axs[1]
    ax.set_box_aspect(3 / 4 / 1.45)
    hi = 0
    for i, (g, light, dark) in enumerate(groups):
        for j, (col, color) in enumerate((('light_noeye_matched_sig', light), ('dark_noeye_sig', dark))):
            y, lo, h, _, _ = frac(col, grp == g)
            x = i + (j - 0.5) * 0.38
            ax.bar(x, y, width=0.34, color=color)
            errbar(ax, x, lo, h)
            hi = max(hi, h)
    ax.axhline(0.05, color=INK2, lw=0.57, ls='--', zorder=0)
    bracket(ax, 0.19, 1.19, 1.06, p[1])
    ax.set_ylim(0, 1.05 * hi)
    ax.set_ylabel('frac. cells with\nsig. encoding')
    leg = legend_above(ax, [Patch(color='#9a9a9a'), Patch(color='#3a3a3a')], ['light', 'dark'], 2, y=1.16)

    for k, (col, ylab) in enumerate((('tuned_dark', 'frac. pillar-tuned\nin dark'),
                                      ('head_centred_dark', 'frac. head-\ncentered in dark'))):
        ax = axs[2 + k]
        ax.set_box_aspect(3 / 4)
        hi = 0
        for i, (g, _, dark) in enumerate(groups):
            y, lo, h, _, _ = frac(col, enough & (grp == g))
            ax.bar(i, y, width=0.6, color=dark)
            errbar(ax, i, lo, h)
            hi = max(hi, h)
        bracket(ax, 0, 1, 1.06, p[2 + k])
        ax.set_ylim(0, 1.05 * hi)
        ax.set_ylabel(ylab)
    for ax, l in zip(axs, letters):
        ax.set_xticks(range(3))
        ax.set_xticklabels(['eye', 'head', 'not\ntuned'])
        ax.set_xlabel(xlabel)
        letter(ax, l, x=-42, level=leg)
    for g, *_ in groups:
        m = grp == g
        print('  {}: n={}, rate ratio {:.3f}, dark sig {:.3f}, tuned dark {:.3f} (n={}), head dark {:.3f}'.format(
            g, int(m.sum()), np.median(ratio[ok_r & m]), frac('dark_noeye_sig', m)[0],
            frac('tuned_dark', enough & m)[0], frac('tuned_dark', enough & m)[4], frac('head_centred_dark', enough & m)[0]))
    print('  Holm p (rate, dark encoding, dark tuning, dark head call):', np.round(p, 3))


def figure9(D):
    """ Are the cells that look head-centred in the light the ones still responsive / tuned in the dark?

    Row 1 groups cells by Figure 1G's model comparison (HEAD vs ADD). Row 2 keeps only cells whose anchoring
    can be measured one by one (notebook X: d' >= 2, called at 10:1), the cell-level counterpart of the
    small true head-centred share; every other tuned cell is left out of row 2's eye / head groups.
    """
    T, _ = cell_table()
    T = T[T['area'].isin(AREAS)].reset_index(drop=True)
    tuned = T['tuned'].to_numpy(bool)
    untuned = np.where(tuned, '', 'untuned')
    call = np.where(T['eye_centred'], 'eye', np.where(T['head_centred'], 'head', untuned))
    anchor = np.where(np.isin(T['anchor_call'], ['eye', 'head']), T['anchor_call'], untuned)
    rng = np.random.default_rng(9)
    fig = plt.figure(figsize=(WIDTH, 5.3), layout='constrained')
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1.45, 1, 1])
    print('Figure 9 row 1 (model-comparison calls):')
    _dark_row(fig, [gs[0, k] for k in range(4)], T, call, 'call in light', 'ABCD', rng)
    print('Figure 9 row 2 (per-cell anchoring, d\' >= 2, 10:1):')
    _dark_row(fig, [gs[1, k] for k in range(4)], T, anchor, 'per-cell anchoring', 'EFGH', rng)
    save(fig, 'Figure9')


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


def area_map(T, values, atlas, k=MAP_K, chunk=1000):
    """ Kernel-smoothed mean of a per-cell value, fit separately in each visual area.

    Each area's map uses only that area's cells and is drawn only inside that
    area's outline, so a value never bleeds across a border (a sign flip
    between neighbouring areas stays a sharp edge); the areas are then laid on
    one image. Every panel covers the same imaged footprint: the smoothing SD
    at each pixel widens (MAP_SIGMA up to MAP_SIGMA_MAX) until it reaches k
    cells that have this value, and pixels that never do are returned as
    'grey' (imaged, but too few cells of this kind). values: (n,) or (n, k).
    """

    pts, shape, ext = map_grid(atlas, MAP_STEP)
    foot = footprint(T, atlas, pts)
    vals = np.asarray(values, float).reshape(len(T), -1)
    M = np.full((len(pts), vals.shape[1]), np.nan)
    for a in AREAS:
        inside = np.flatnonzero(Path(atlas[a]).contains_points(pts) & foot)
        m = (T['area'].to_numpy() == a) & np.all(np.isfinite(vals), 1)
        if not len(inside) or not m.any():
            continue
        cxy = T.loc[m, ['x', 'y']].to_numpy(float)
        for i in range(0, len(inside), chunk):
            ix = inside[i:i + chunk]
            w, ok = _adaptive(pts[ix], cxy, k, MAP_SIGMA, MAP_SIGMA_MAX)
            v = w @ vals[m] / w.sum(1, keepdims=True)
            M[ix[ok]] = v[ok]
    grey = foot & ~np.all(np.isfinite(M), 1)
    return M.reshape(shape + (vals.shape[1],)), grey.reshape(shape), ext


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


def draw_map(fig, ax, M, grey, ext, atlas, cmap, label, sym=False, lims=None, ticks=None, names=False):
    from matplotlib.colors import ListedColormap, Normalize
    v = M[np.isfinite(M)]
    if lims is None:
        lims = ((-np.percentile(np.abs(v), 98), np.percentile(np.abs(v), 98)) if sym else
                tuple(np.percentile(v, [2, 98])))
    ax.imshow(np.ma.masked_where(~grey, np.ones(grey.shape)), extent=ext, cmap=ListedColormap([NO_DATA]),
              interpolation='nearest')
    im = ax.imshow(np.ma.masked_invalid(M), extent=ext, cmap=cmap, norm=Normalize(*lims), interpolation='nearest')
    for k, c in atlas.items():
        ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=INK2 if k in AREAS else '#b9b8b2', lw=0.6)
        if names and k in AREAS:
            ax.text(*c.mean(0), k, ha='center', va='center', fontsize=FONT, color=INK)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect('equal')
    ax.axis('off')
    cb = fig.colorbar(im, ax=ax, shrink=0.75, aspect=14, pad=0.02, ticks=ticks)
    cb.set_label(label, fontsize=FONT)
    cb.ax.tick_params(labelsize=FONT, width=0.45, length=1.7)
    cb.outline.set_linewidth(0.45)
    return im


def no_data_legend(ax):
    legend_above(ax, [Patch(color=NO_DATA)], ['imaged, too few cells of this kind'], 1, y=1.0)


def compass(ax):
    kw = dict(xycoords='axes fraction', textcoords='axes fraction', fontsize=FONT, color=INK2,
              arrowprops=dict(arrowstyle='<-', color=INK2, lw=0.8))
    ax.annotate('anterior', xy=(0.04, 0.06), xytext=(0.04, 0.24), ha='center', va='bottom', **kw)
    ax.annotate('medial', xy=(0.04, 0.06), xytext=(0.24, 0.06), ha='left', va='center', **kw)


def fig10_values(T):
    """ Figure 10 A-F: one value per cell (NaN = not of the kind mapped), colour map, label, symmetric limits. """
    eye_head = two_pole(HEAD, EYE)
    seq = 'Blues'
    tuned = T['tuned'].to_numpy(bool)
    call = np.where(T['eye_centred'], 1.0, np.where(T['head_centred'], -1.0, 0.0))
    gain = T['gain_cell'].to_numpy(float)
    sign = np.where(gain == 1, np.sign(T['k'].to_numpy(float)), np.nan)
    return [
        (tuned.astype(float), seq, 'frac. pillar-tuned', False),
        (np.where(tuned, call, np.nan), eye_head, 'eye- minus head-\ncentered (frac. tuned)', True),
        (T['field_shift'].to_numpy(float), eye_head, 'field-shift index', True),
        (np.where(T['q_row'] >= 0, gain, np.nan), seq, 'frac. gain cells', False),
        (sign, two_pole(DOWN, UP), 'gain up minus down\nwith nasal eye', True),
        (T['log2_dark_light'].to_numpy(float), seq, 'log2 dark / light rate', False),
    ]


def figure10(D):
    """ Top-down cortical maps of the main per-cell variables, all animals pooled. """
    T, Q = cell_table()
    atlas = load_atlas()
    T = map_cells(T, atlas)
    seq = 'Blues'
    maps = fig10_values(T)
    fig = plt.figure(figsize=(WIDTH, 10.6), layout='constrained')
    gs = fig.add_gridspec(4, 3)
    for i, (v, cmap, label, sym) in enumerate(maps):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        M, grey, ext = area_map(T, v, atlas)
        draw_map(fig, ax, M[..., 0], grey, ext, atlas, cmap, label, sym=sym, names=i == 0)
        if i == 0:
            compass(ax)
            no_data_legend(ax)
        letter(ax, 'ABCDEF'[i], x=0, y=0)
    for r, (cols, where) in enumerate(((Q['qvis'], 'visible'), (~Q['qvis'], 'never visible'))):
        G, ext = local_gain_maps(T, Q, atlas, cols)
        for c, (key, label) in enumerate((('need', 'gain signal (x100)'), ('coverage', 'coverage'),
                                          ('eff_dims', 'effective gain\ndimensions'))):
            ax = fig.add_subplot(gs[2 + r, c])
            draw_map(fig, ax, *G[key], ext, atlas, seq, '{}:\n{}'.format(where, label))
            letter(ax, 'GHIJKL'[3 * r + c], x=0, y=0)
    save(fig, 'Figure10')


def figure11(D):
    """ Horizontal retinotopic preference: preferred retinal azimuth of the pillar, smoothed as a circular mean. """
    T, _ = cell_table()
    atlas = load_atlas()
    T = map_cells(T, atlas)
    pref = np.radians(T['pref_azimuth'].to_numpy(float))
    unit = np.c_[np.cos(pref), np.sin(pref)]
    tuned = T['tuned'].to_numpy(bool)
    fig = plt.figure(figsize=(WIDTH, 3.3), layout='constrained')
    gs = fig.add_gridspec(1, 3)
    ticks = [-180, -90, 0, 90, 180]
    for i, (m, label) in enumerate(((np.ones(len(T), bool), 'all cells:\npref. retinal azimuth (deg)'),
                                    (tuned, 'pillar-tuned cells:\npref. retinal azimuth (deg)'))):
        M, grey, ext = area_map(T, np.where(m[:, None], unit, np.nan), atlas)
        ax = fig.add_subplot(gs[i])
        draw_map(fig, ax, np.degrees(np.arctan2(M[..., 1], M[..., 0])), grey, ext, atlas, 'twilight_shifted', label,
                 lims=(-180, 180), ticks=ticks, names=i == 0)
        if i == 0:
            compass(ax)
            no_data_legend(ax)
        letter(ax, 'AB'[i], x=0, y=0)
    M, grey, ext = area_map(T, np.where(tuned[:, None], unit, np.nan), atlas)
    ax = fig.add_subplot(gs[2])
    draw_map(fig, ax, np.hypot(M[..., 0], M[..., 1]), grey, ext, atlas, 'Blues', 'pillar-tuned cells:\nagreement of preferences',
             lims=(0, None))
    letter(ax, 'C', x=0, y=0)
    save(fig, 'Figure11')


# ---------------------------------------------------------------------------
# Figures 10 and 11 redrawn with one dot per cell instead of smoothed maps
# (saved as Figure10_cells / Figure11_cells; the map figures are unchanged).
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


def figure10_cells(D):
    """ Figure 10 with every cell as a dot coloured by its own value, in place of the smoothed maps.

    A-F and G, J are per-cell values. Coverage and effective dimensions (H, I, K, L) exist only for a
    population, so there each cell with a gain profile is coloured by the local-population value at its
    position (the map sampled at the cell). Continuous colours are clipped at the cells' 2nd / 98th percentile.
    """
    T, Q = cell_table()
    atlas = load_atlas()
    T = map_cells(T, atlas)
    bl = plt.get_cmap('Blues')
    lo_b, hi_b = bl(0.3), bl(0.85)
    cats = {0: [(0, lo_b, 'not tuned'), (1, hi_b, 'pillar-tuned')],
            1: [(-1, HEAD, 'head-centered'), (0, SCATTER_MID, 'tuned, neither'), (1, EYE, 'eye-centered')],
            3: [(0, lo_b, 'no gain'), (1, hi_b, 'gain cell')],
            4: [(1, UP, 'gain up, nasal eye'), (-1, DOWN, 'gain down')]}
    cmaps = {2: two_pole(HEAD, EYE, SCATTER_MID), 5: blues_from()}
    xy = T[['x', 'y']].to_numpy(float)
    fig = plt.figure(figsize=(WIDTH, 11.4), layout='constrained')
    gs = fig.add_gridspec(4, 3)
    for i, (v, _, label, sym) in enumerate(fig10_values(T)):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        if i in cats:
            cell_scatter(fig, ax, T, v, atlas, cats=cats[i], names=i == 0)
        else:
            cell_scatter(fig, ax, T, v, atlas, cmaps[i], label, lims=_lims(v, sym), extend='both')
        if i == 0:
            compass(ax)
        letter(ax, 'ABCDEF'[i], x=0, y=0)
    profiled = (T['q_row'] >= 0).to_numpy()
    for r, (cols, where) in enumerate(((Q['qvis'], 'visible'), (~Q['qvis'], 'never visible'))):
        G, ext = local_gain_maps(T, Q, atlas, cols)
        need = cell_gain_signal(T, Q, cols)
        for c, (key, label) in enumerate((('need', 'gain signal (x100)'), ('coverage', 'coverage'),
                                          ('eff_dims', 'effective gain\ndimensions'))):
            ax = fig.add_subplot(gs[2 + r, c])
            if key == 'need':
                cell_scatter(fig, ax, T, need, atlas, blues_from(), '{}:\n{}, per cell'.format(where, label),
                             lims=_lims(need, False), extend='both')
            else:
                v = np.where(profiled, sample_map(G[key][0], ext, xy), np.nan)
                cell_scatter(fig, ax, T, v, atlas, blues_from(), '{}: {}\n(local population)'.format(where, label),
                             lims=_lims(G[key][0], False), extend='both', missing='no value here / no profile')
            letter(ax, 'GHIJKL'[3 * r + c], x=0, y=0)
    save(fig, 'Figure10_cells')


def figure11_cells(D):
    """ Figure 11 with every cell as a dot: A, B each cell's own preferred retinal azimuth; C (agreement, a
    population quantity) is the map's value sampled at each tuned cell. """
    T, _ = cell_table()
    atlas = load_atlas()
    T = map_cells(T, atlas)
    pref = T['pref_azimuth'].to_numpy(float)
    tuned = T['tuned'].to_numpy(bool)
    fig = plt.figure(figsize=(WIDTH, 3.6), layout='constrained')
    gs = fig.add_gridspec(1, 3)
    ticks = [-180, -90, 0, 90, 180]
    for i, (v, label) in enumerate(((pref, 'all cells:\npref. retinal azimuth (deg)'),
                                    (np.where(tuned, pref, np.nan), 'pillar-tuned cells:\npref. retinal azimuth (deg)'))):
        ax = fig.add_subplot(gs[i])
        cell_scatter(fig, ax, T, v, atlas, 'twilight_shifted', label, lims=(-180, 180), ticks=ticks, names=i == 0)
        if i == 0:
            compass(ax)
        letter(ax, 'AB'[i], x=0, y=0)
    rad = np.radians(pref)
    M, grey, ext = area_map(T, np.where(tuned[:, None], np.c_[np.cos(rad), np.sin(rad)], np.nan), atlas)
    agree = np.hypot(M[..., 0], M[..., 1])
    v = np.where(tuned, sample_map(agree, ext, T[['x', 'y']].to_numpy(float)), np.nan)
    ax = fig.add_subplot(gs[2])
    cell_scatter(fig, ax, T, v, atlas, blues_from(), 'pillar-tuned cells: agreement\nof preferences (local population)',
                 lims=(0, np.nanmax(agree)), missing='not tuned / no value here')
    letter(ax, 'C', x=0, y=0)
    save(fig, 'Figure11_cells')


# ---------------------------------------------------------------------------
# Figures 12-14: is the eye-position gain a left-over gaze-shift transient?
# Computed by gaze_shift_controls.py (cached per session in manuscript_figures/gaze_controls/;
# run `python gaze_shift_controls.py` first, `--force` to recompute).
# ---------------------------------------------------------------------------

EARLY_C, LATE_C, SHIFT_C = '#f3c98b', '#8fb8de', '#d9d8d3'


def _gsc():
    import gaze_shift_controls as gsc
    return gsc


def pop_peth(P, base):
    """ Pooled fractional change from baseline over cells: sum_c (P_c - b_c) / sum_c b_c. """
    b = np.nanmean(P[:, base], 1)
    ok = np.isfinite(b) & (b > 0)
    return np.nansum(P[ok] - b[ok, None], 0) / np.nansum(b[ok])


def boot_frac(k, n):
    return wilson(k, n)


def example_window(gsc, sess_key='DMM061_pos18', width=20.0):
    """ One light stretch: eye position, gaze-shift onsets, epoch labels and the responsive cells' activity. """
    neural, nonvis, sess, animal = [s for s in gsc.sessions() if sess_key in s[2]][0]
    D, ev, cells, lag = gsc.load(neural, nonvis)
    E = gsc.frame_epochs(D, ev)
    M = gsc.epoch_masks(D, E)
    r = gsc.results()[sess]['transients']
    t = D['twopT']
    starts = np.arange(t[0], t[-1] - width, 2.0)
    best, score = starts[0], -1
    for s0 in starts:
        w = (t >= s0) & (t < s0 + width)
        if D['light'][w].mean() < 1 or np.isfinite(D['e'][w]).mean() < 0.95:
            continue
        n_ev = np.sum((ev['t_on'] >= s0) & (ev['t_on'] < s0 + width))
        # an active stretch: late-fixation frames that sit between gaze shifts (short fixations), no long rest
        if np.any(M['late'][w] & ~M['late_act'][w]):
            continue
        sc = min(n_ev, 25) * M['late_act'][w].sum()
        if sc > score:
            best, score = s0, sc
    w = (t >= best) & (t < best + width)
    ep = np.where(E['inside'], 0, np.where(M['early'], 1, np.where(M['late'], 3, 2)))
    print('Figure 12A: {} from {:.0f} s, {} gaze shifts, {} late-fixation frames'.format(
        sess, best, len(ev['t_on'][(ev['t_on'] >= best) & (ev['t_on'] < best + width)]), int(M['late'][w].sum())))
    return {'t': t[w] - best, 'e': D['e'][w], 'ep': ep[w],
            'on': ev['t_on'][(ev['t_on'] >= best) & (ev['t_on'] < best + width)] - best, 'dt': D['dt']}


STATE_CACHE = os.path.join(HERE, 'manuscript_figures', 'gaze_controls_state.npz')
FIX_BINS = np.r_[0.25, np.geomspace(0.5, 300, 40)]


def fixation_state(gsc):
    """ Fixation duration of every early / late frame, and speed, pooled over sessions. Cached. """
    if os.path.isfile(STATE_CACHE):
        return dict(np.load(STATE_CACHE))
    out = {k: np.zeros(len(FIX_BINS) - 1) for k in ('early', 'late')}
    for neural, nonvis, sess, animal in gsc.sessions():
        Dd, ev, cells, lag = gsc.load(neural, nonvis)
        E = gsc.frame_epochs(Dd, ev)
        M = gsc.epoch_masks(Dd, E)
        for k in out:
            out[k] += np.histogram(np.clip(E['fix_dur'][M[k]], FIX_BINS[0], FIX_BINS[-1] - 1e-9), FIX_BINS)[0]
    np.savez(STATE_CACHE, **out)
    return out


EVENT_CACHE = os.path.join(HERE, 'manuscript_figures', 'gaze_controls_events.npz')
EVENT_EDGES = np.arange(-1.0, 2.0 + 1e-9, 2 / 15)        # 2P frame bins (133 ms) around gaze-shift onset


def event_matrix(gsc):
    """ Population activity (mean over all cells, z-scored on each session's light frames) in frame bins
    around every light gaze shift with no other shift in the 0.8 s before; and the time to the next shift. Only
    events with eye tracking throughout the 0.8 s before and from onset to the next shift (or the 2 s plotted),
    so no shift hidden in a tracking gap goes uncounted. Cached. """
    if os.path.isfile(EVENT_CACHE):
        return dict(np.load(EVENT_CACHE))
    rows, nxt = [], []
    for neural, nonvis, sess, animal in gsc.sessions():
        Dd, ev, cells, lag = gsc.load(neural, nonvis)
        use = Dd['use'] & Dd['light']
        pop = np.where(use, Dd['Y'].mean(1), np.nan)
        z = (pop - np.nanmean(pop)) / np.nanstd(pop)
        o = np.argsort(ev['t_on'])
        on, lt = ev['t_on'][o], ev['light'][o]
        gap_pre, gap_post = on - np.r_[-np.inf, on[:-1]], np.r_[on[1:], np.inf] - on
        T = Dd['twopT']
        cov, clean = ev['cover_pre'][o], ev['clean_until'][o]
        ok = lt & (gap_pre > gsc.ISO_PRE) & cov & (clean >= np.minimum(gap_post, EVENT_EDGES[-1]))
        for k in np.flatnonzero(ok):
            rel = T - on[k]
            m = (rel >= EVENT_EDGES[0]) & (rel < EVENT_EDGES[-1])
            r = np.full(len(EVENT_EDGES) - 1, np.nan)
            r[np.digitize(rel[m], EVENT_EDGES) - 1] = z[m]
            rows.append(r)
            nxt.append(gap_post[k])
    out = {'R': np.array(rows), 'next': np.array(nxt)}
    np.savez(EVENT_CACHE, **out)
    return out


def figure12(D):
    """ Gaze shifts and their after-effects are resolved, and late-fixation frames are free of them. """
    gsc = _gsc()
    T, R = gsc.pooled()
    fig = plt.figure(figsize=(WIDTH, 7.4), layout='constrained')
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 2.4, 2.0])
    mid = gs[1].subgridspec(1, 2, width_ratios=[1.3, 1])
    bot = gs[2].subgridspec(1, 3)

    X = example_window(gsc)
    ax = fig.add_subplot(gs[0])
    cols = {0: SHIFT_C, 1: EARLY_C, 3: LATE_C}
    for k, c in cols.items():
        m = X['ep'] == k
        for a, b in zip(np.flatnonzero(np.diff(np.r_[0, m.astype(int)]) == 1),
                        np.flatnonzero(np.diff(np.r_[m.astype(int), 0]) == -1)):
            ax.axvspan(X['t'][a] - X['dt'] / 2, X['t'][b] + X['dt'] / 2, color=c, lw=0, zorder=0)
    ax.plot(X['t'], X['e'], color=INK, lw=1.0)
    for x in X['on']:
        ax.axvline(x, color=INK2, lw=0.5, ls=':', zorder=1)
    ax.set_xlim(0, X['t'][-1])
    ax.set_ylabel('eye in head\n(deg)')
    ax.set_xlabel('time (s)')
    legend_above(ax, [Patch(color=SHIFT_C), Patch(color=EARLY_C), Patch(color=LATE_C),
                      Line2D([], [], color=INK2, lw=0.8, ls=':')],
                 ['frame holds a gaze shift', 'early fixation', 'late fixation', 'gaze-shift onset'], 4)
    letter(ax, 'A')

    Ev = event_matrix(gsc)
    o = np.argsort(Ev['next'])
    M_, nxt = Ev['R'][o], Ev['next'][o]
    cen = (EVENT_EDGES[:-1] + EVENT_EDGES[1:]) / 2
    M_ = M_ - np.nanmean(M_[:, (cen >= -0.8) & (cen < -0.1)], 1, keepdims=True)    # each event's own pre-shift level
    W = 150                                                 # running mean over neighbouring events, display only
    Ms = np.array([np.nanmean(M_[max(0, i - W // 2):i + W // 2], 0) for i in range(len(M_))])
    ax = fig.add_subplot(mid[0])
    im = ax.imshow(Ms, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3, interpolation='nearest',
                   extent=(EVENT_EDGES[0], EVENT_EDGES[-1], len(Ms), 0))
    ax.plot(np.clip(nxt, None, EVENT_EDGES[-1]), np.arange(len(nxt)), color=INK, lw=1.2)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel('time from gaze-shift onset (s)')
    ax.set_ylabel('gaze shifts, sorted by\ntime to the next one')
    ax.set_yticks([0, len(Ms)])
    cb = fig.colorbar(im, ax=ax, shrink=0.8, aspect=16, pad=0.02)
    cb.set_label('activity change from\npre-shift level (z)', fontsize=FONT)
    cb.ax.tick_params(labelsize=FONT, width=0.45, length=1.7)
    cb.outline.set_linewidth(0.45)
    legend_above(ax, [Line2D([], [], color=INK, lw=1.2)], ['next gaze shift'], 1)
    letter(ax, 'B')

    ax = fig.add_subplot(mid[1])
    ax.set_box_aspect(3 / 4)
    vir = plt.get_cmap('viridis')
    groups = ((0, 0.5, vir(0.85)), (0.5, 1, vir(0.6)), (1, 2, vir(0.33)), (2, np.inf, vir(0.05)))
    for a, b, c in groups:
        m = (nxt >= a) & (nxt < b)
        y = np.nanmean(M_[m], 0)
        se = np.nanstd(M_[m], 0) / np.sqrt(np.sum(np.isfinite(M_[m]), 0))
        ax.fill_between(cen, y - 1.96 * se, y + 1.96 * se, color=c, alpha=0.25, lw=0)
        ax.plot(cen, y, color=c, lw=1.4)
        pk = np.argmax(np.where(cen > 0, y, -np.inf))
        print('Figure 12C next shift {}-{} s: {} events, peak {:.2f} z at {:.0f} ms'.format(a, b, int(m.sum()), y[pk], 1000 * cen[pk]))
    ax.axvline(0, color=INK2, lw=0.4)
    ax.axvspan(gsc.LATE_S + 0.1, EVENT_EDGES[-1], color=LATE_C, alpha=0.25, lw=0, zorder=0)
    ax.set_xlim(EVENT_EDGES[0], EVENT_EDGES[-1])
    ax.set_xlabel('time from gaze-shift onset (s)')
    ax.axhline(0, color=INK2, lw=0.4)
    ax.set_ylabel('activity change from\npre-shift level (z)')
    legend_above(ax, [Line2D([], [], color=c, lw=1.4) for *_, c in groups],
                 ['next shift < 0.5 s', '0.5-1 s', '1-2 s', '> 2 s'], 2, title='')
    letter(ax, 'C')
    print('Figure 12B: {} isolated light gaze shifts'.format(len(nxt)))

    ax = fig.add_subplot(bot[0])
    ax.set_box_aspect(3 / 4)
    hi = 0
    for i, a in enumerate(AREAS):
        m = T['area'] == a
        hi = max(hi, _frac_bars(ax, [i - 0.17], [(int(T['kernels/kernel_call'][m].sum()), int(m.sum()))], [AREA_COLOR[a]], 0.32))
        hi = max(hi, _frac_bars(ax, [i + 0.17], [_null_frac(T['kernels/kernel_null_call'], m)], [SHUF], 0.32))
        print('Figure 12D: {} {:.3f} of {} cells predicted better with gaze-shift kernels (shift times moved: {:.3f})'.format(
            a, T['kernels/kernel_call'][m].mean(), int(m.sum()), T['kernels/kernel_null_call'][m].mean()))
    area_ticks(ax)
    ax.set_ylim(0, 1.05 * hi)
    ax.set_ylabel('frac. cells tracking\ngaze shifts')
    legend_above(ax, [Patch(color='#9a9a9a'), Patch(color=SHUF)], ['real shift times', 'shift times moved'], 1)
    letter(ax, 'D')

    ax = fig.add_subplot(bot[1])
    ax.set_box_aspect(3 / 4)
    edges = np.arange(-40, 41, 2)
    x = (edges[:-1] + edges[1:]) / 2
    for k, c in (('early', '#d08a1e'), ('late', '#2a6fb0')):
        h = np.sum([r['e_hist'][k] for r in R.values()], 0).astype(float)
        ax.plot(x, h / h.sum(), color=c, lw=1.4)
    ax.set_xlabel('eye in head (deg from median)')
    ax.set_ylabel('frac. frames')
    legend_above(ax, [Line2D([], [], color='#d08a1e', lw=1.4), Line2D([], [], color='#2a6fb0', lw=1.4)],
                 ['early fixation', 'late fixation'], 1)
    letter(ax, 'E')

    ax = fig.add_subplot(bot[2])
    ax.set_box_aspect(3 / 4)
    F = fixation_state(gsc)
    for k, c in (('early', '#d08a1e'), ('late', '#2a6fb0')):
        cum = np.cumsum(F[k]) / F[k].sum()
        ax.step(FIX_BINS[1:], cum, where='post', color=c, lw=1.4)
        print('Figure 12F: {} frames: {:.2f} in fixations < 3 s, {:.2f} in fixations > 10 s'.format(
            k, np.interp(3.0, FIX_BINS[1:], cum), 1 - np.interp(10.0, FIX_BINS[1:], cum)))
    ax.axvline(3.0, color=INK2, lw=0.57, ls='--')
    ax.set_xscale('log')
    ax.set_xticks([1, 3, 10, 100])
    ax.set_xticklabels(['1', '3', '10', '100'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, 1)
    ax.set_xlabel('fixation duration (s)')
    ax.set_ylabel('cumulative frac.\nframes')
    legend_above(ax, [Line2D([], [], color='#d08a1e', lw=1.4), Line2D([], [], color='#2a6fb0', lw=1.4)],
                 ['early fixation', 'late fixation'], 1)
    letter(ax, 'F')
    save(fig, 'Figure12')


def _frac_bars(ax, xs, vals, colors, width, hatch=None):
    """ Bars of fractions with Wilson 95% intervals. vals = [(k, n), ...]. """
    hi = 0
    for x, (k, n), c in zip(xs, vals, colors):
        y, lo, h = wilson(k, n)
        ax.bar(x, y, width=width, color=c, hatch=hatch, edgecolor=INK2 if hatch else 'none', lw=0.28)
        errbar(ax, x, lo, h)
        hi = max(hi, h)
    return hi


def _null_frac(null_calls, m):
    """ Mean null call rate over shuffles (cells x shuffles), as (k, n) with n = cells. """
    v = null_calls[m].mean(1)
    return int(round(v.sum())), int(m.sum())


N_SHUF_GSC = 10
BEH_LABEL = {'shift': 'gaze\nshift', 'post': '< 0.3 s\nafter shift', 'comp': 'compen-\nsatory', 'still': 'still',
             'all': 'all light\nframes'}


EXAMPLES_13 = [('251016_DMM_DMM061_pos18', 86), ('251101_DMM_DMM056_pos07', 4)]   # cells 11 and 12
EXAMPLE_CACHE = os.path.join(HERE, 'manuscript_figures', 'gaze_controls_examples.npz')


def example_data(gsc):
    """ Retinal tuning at two eye positions for the example cells, all light frames (cached). """
    if os.path.isfile(EXAMPLE_CACHE):
        z = np.load(EXAMPLE_CACHE)
        return [{k[len('{}_'.format(i)):]: z[k] for k in z.files if k.startswith('{}_'.format(i))} for i in range(len(EXAMPLES_13))]
    S = {x[2]: x for x in gsc.sessions()}
    out = [gsc.example_all_frames(S[sess][0], S[sess][1], c) for sess, c in EXAMPLES_13]
    np.savez(EXAMPLE_CACHE, **{'{}_{}'.format(i, k): v for i, o in enumerate(out) for k, v in o.items()})
    return out


def example_panel(ax, o, area, num, lab_y):
    cen = (o['edges'][:-1] + o['edges'][1:]) / 2
    for h, ls, mk, mfc in ((0, LS_LOW, 'o', INK), (1, LS_HIGH, 's', 'white')):
        m = np.isfinite(o['meas'][h])
        ax.plot(cen[m], o['add'][h][m], color='#9a9a9a', ls=ls, lw=1.0, zorder=2)
        ax.plot(cen[m], o['gain'][h][m], color=EYE, ls=ls, lw=1.6, zorder=3)
        ax.errorbar(cen[m], o['meas'][h][m], yerr=o['sem'][h][m], fmt=mk, ms=3.6, color=INK, mfc=mfc, mew=0.9,
                    elinewidth=0.9, capsize=0, zorder=4)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel('retinal position (deg)')
    if lab_y:
        ax.set_ylabel('activity (norm. spikes)')
    ax.text(0.03, 0.97, '{}, cell {}'.format(area, num), transform=ax.transAxes, ha='left', va='top', fontsize=FONT)


ARR_TOWARD, ARR_AWAY = '#52514e', '#b9b8b2'


def arrival_cells(R, k, sel='gain'):
    """ Per-cell arrival-test rates pooled over sessions: (cells, eye side, shift direction) for in and out of
    the field, keeping cells with every group filled. """
    out = {}
    for fld in ('in', 'out'):
        X = np.concatenate([r['arrival'][k]['rate_' + fld][r['cells'][sel].astype(bool)] for r in R.values()])
        out[fld] = X[np.all(np.isfinite(X.reshape(len(X), -1)), 1)]
    return out


def arrival_effects(X, rng, n_boot=2000):
    """ Eye-side and arrival effects (mean over the other factor) with bootstrap-over-cells 95% CIs. """
    eye = (X[:, 1, :] - X[:, 0, :]).mean(1)
    arr = (X[:, :, 1] - X[:, :, 0]).mean(1)
    res = {}
    for nm, v in (('eye', eye), ('arrival', arr)):
        b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
        res[nm] = (v.mean(), *np.percentile(b, [2.5, 97.5]))
    return res


def arrival_panel(ax, gsc, R, k, legend=False, title=None):
    """ Rate (x the cell's mean) with the eye away from / toward each cell's high-gain side, split by the direction
    of the last gaze shift; filled = pillar in the field, open = outside it. Means over cells, 95% bootstrap CI. """
    rng = np.random.default_rng(0)
    X = arrival_cells(R, k)
    hi = 0
    mids = {}
    for fld, mk, ls in (('in', 'o', '-'), ('out', 'o', '--')):
        Z = X[fld]
        mids[fld] = Z.mean(0).mean()
        for b, (c, off) in enumerate(((ARR_AWAY, -0.04), (ARR_TOWARD, 0.04))):
            m = Z[:, :, b].mean(0)
            bs = np.array([Z[rng.integers(0, len(Z), len(Z)), :, b].mean(0) for _ in range(2000)])
            lo_, hi_ = np.percentile(bs, [2.5, 97.5], 0)
            ax.errorbar(np.array([0, 1]) + off, m, yerr=[m - lo_, hi_ - m], color=c, marker=mk, ls=ls, ms=3.6, lw=1.2,
                        elinewidth=0.9, capsize=0, mfc=c if fld == 'in' else 'white', mec=c)
            hi = max(hi, np.max(hi_))
        e = arrival_effects(Z, rng)
        print('Figure 13G {} {} field, {} gain cells: eye effect {:+.3f} [{:+.3f}, {:+.3f}], arrival effect {:+.3f} '
              '[{:+.3f}, {:+.3f}]'.format(k, fld, len(Z), *e['eye'], *e['arrival']))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['away from', 'toward'])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, 1.08 * hi)
    ax.text(0.5, mids['in'] + 0.12 * hi, 'in field', ha='center', va='bottom', fontsize=FONT - 3, color=INK2)
    ax.text(0.5, mids['out'] - 0.06 * hi, 'outside', ha='center', va='top', fontsize=FONT - 3, color=INK2)
    ax.set_xlabel("eye position vs. cell's\nhigh-gain side" if title is None else title)
    ax.set_ylabel('activity (x mean rate)')
    if legend:
        legend_above(ax, [Line2D([], [], color=ARR_TOWARD, lw=1.4), Line2D([], [], color=ARR_AWAY, lw=1.4)],
                     ['last shift toward', 'last shift away'], 1)


SINCE_LABELS = ['0-0.3', '0.3-1', '1-3', '3-10', '>10']


def arrival_since_rates(R, suf='', sel='gain'):
    """ Per bin of time since the last shift: in-field rates (cells, eye away / toward), averaged over the last
    shift's direction where both eye sides are filled. """
    nb = len(R[next(iter(R))]['arrival']['meta']['since_edges']) - 1
    out = []
    for i in range(nb):
        X = np.concatenate([r['arrival']['since_{}{}'.format(i, suf)]['rate_in'][r['cells'][sel].astype(bool)]
                            for r in R.values()])
        with np.errstate(invalid='ignore'):
            Xe = np.nanmean(X, 2)
        out.append(Xe[np.all(np.isfinite(Xe), 1)])
    return out


def _boot_mean(v, rng, n=2000):
    b = np.array([v[rng.integers(0, len(v), len(v))].mean(0) for _ in range(n)])
    return np.percentile(b, [2.5, 97.5], 0)


def arrival_since_panel(ax, gsc, R):
    """ In-field activity with the eye toward / away from each cell's high-gain side, by time since the last gaze
    shift; means over gain cells, 95% bootstrap CI. """
    rng = np.random.default_rng(1)
    Xs = arrival_since_rates(R)
    x = np.arange(len(Xs))
    hi = 0
    for j, (lab, mfc, off) in enumerate((('eye away', 'white', -0.06), ('eye toward', INK, 0.06))):
        m = np.array([X[:, j].mean() for X in Xs])
        ci = np.array([_boot_mean(X[:, j], rng) for X in Xs])
        ax.errorbar(x + off, m, yerr=[m - ci[:, 0], ci[:, 1] - m], color=INK, marker='o', mfc=mfc, mec=INK, ms=3.6,
                    lw=1.2, elinewidth=0.9, capsize=0, ls='--' if j == 0 else '-')
        hi = max(hi, ci[:, 1].max())
    for i, X in enumerate(Xs):
        d = X[:, 1] - X[:, 0]
        lo_, hi_ = _boot_mean(d, rng)
        print('Figure 13_arrival D since {} s: {} gain cells, eye away {:.2f}, toward {:.2f}, difference {:+.2f} [{:+.2f}, {:+.2f}]'
              .format(SINCE_LABELS[i], len(X), X[:, 0].mean(), X[:, 1].mean(), d.mean(), lo_, hi_))
    ax.set_xticks(x)
    ax.set_xticklabels(SINCE_LABELS)
    ax.set_xlim(-0.4, len(Xs) - 0.6)
    ax.set_ylim(0, 1.08 * hi)
    ax.set_xlabel('time since last gaze shift (s)')
    ax.set_ylabel('activity, pillar in field\n(x mean rate)')
    legend_above(ax, [Line2D([], [], color=INK, marker='o', ms=3.6, ls='-'),
                      Line2D([], [], color=INK, marker='o', mfc='white', ms=3.6, ls='--')],
                 ['eye toward high-gain side', 'eye away from it'], 1)


def arrival_since_diffs(R, suf='', fld='in', sel='gain', matched=True):
    """ Per bin of time since the last shift: per-cell eye toward - away difference (x mean rate), matched for
    distance from the median eye position unless matched=False. """
    nb = len(R[next(iter(R))]['arrival']['meta']['since_edges']) - 1
    key = 'diff_{}_{}'.format(fld, 'm' if matched else 'u')
    out = []
    for i in range(nb):
        v = np.concatenate([r['arrival']['since_{}{}'.format(i, suf)][key][r['cells'][sel].astype(bool)]
                            for r in R.values()])
        out.append(v[np.isfinite(v)])
    return out


def _diff_line(ax, Vs, x, c, rng, lab, mfc=None, min_cells=20):
    m, lo, hi_ = np.full(len(Vs), np.nan), np.full(len(Vs), np.nan), np.full(len(Vs), np.nan)
    for i, v in enumerate(Vs):
        if len(v) < min_cells:
            continue
        m[i] = v.mean()
        lo[i], hi_[i] = _boot_mean(v, rng)
        print('  {} since {} s: {} cells, {:+.2f} [{:+.2f}, {:+.2f}]'.format(lab, SINCE_LABELS[i], len(v), m[i], lo[i], hi_[i]))
    ax.errorbar(x, m, yerr=[m - lo, hi_ - m], color=c, marker='o', ms=3.6, lw=1.2, elinewidth=0.9, capsize=0,
                mfc=mfc or c, mec=c)
    return np.nanmax(hi_), np.nanmin(lo)


def arrival_since_matched_panel(ax, gsc, R):
    """ Figure 13C: eye-position effect (toward - away from each gain cell's high-gain side, matched for distance
    from the median eye position) by time since the last gaze shift, pillar in vs outside the field. """
    rng = np.random.default_rng(3)
    x = np.arange(len(SINCE_LABELS))
    print('Figure 13C (gain cells, matched eye-side difference, x mean rate):')
    hi1, lo1 = _diff_line(ax, arrival_since_diffs(R, fld='in'), x - 0.05, INK, rng, 'in field')
    hi2, lo2 = _diff_line(ax, arrival_since_diffs(R, fld='out'), x + 0.05, '#b9b8b2', rng, 'outside', mfc='white')
    ax.axhline(0, color=INK2, lw=0.57, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(SINCE_LABELS)
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.set_ylim(min(lo1, lo2, 0) * 1.1 - 0.05, 1.08 * max(hi1, hi2))
    ax.set_xlabel('time since last gaze shift (s)')
    ax.set_ylabel('eye toward - away\n(x mean rate)')
    legend_above(ax, [Line2D([], [], color=INK, marker='o', ms=3.6),
                      Line2D([], [], color='#b9b8b2', marker='o', mfc='white', ms=3.6)],
                 ['pillar in field', 'outside it'], 2)


def arrival_since_diff_panel(ax, gsc, R):
    """ Matched eye-side difference by time since the last shift, all / moving / still frames (pillar in field). """
    rng = np.random.default_rng(2)
    handles, labels = [], []
    x = np.arange(len(SINCE_LABELS))
    for suf, c, lab, off in (('', INK, 'all frames', 0.0), ('_move', EYE, 'moving', -0.1), ('_still', '#b9b8b2', 'still', 0.1)):
        print('Figure 13_arrival E ' + lab + ':')
        _diff_line(ax, arrival_since_diffs(R, suf), x + off, c, rng, lab)
        handles.append(Line2D([], [], color=c, marker='o', ms=3.6))
        labels.append(lab)
    ax.axhline(0, color=INK2, lw=0.57, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(SINCE_LABELS)
    ax.set_xlabel('time since last gaze shift (s)')
    ax.set_ylabel('eye toward - away\n(x mean rate, pillar in field)')
    legend_above(ax, handles, labels, 3)


def figure13_arrival(D):
    """ Supplement to Figure 13C and G: the arrival test on settled, still and compensatory frames (A-C); in-field
    activity by time since the last shift, eye toward vs away (D); the matched eye-side difference on all, moving
    and still frames (E). """
    gsc = _gsc()
    R = gsc.results()
    fig = plt.figure(figsize=(WIDTH, 7.4), layout='constrained')
    gs = fig.add_gridspec(2, 3)
    for j, (k, lab) in enumerate((('settled', 'settled frames'), ('still', 'still'), ('comp', 'compensatory head turns'))):
        ax = fig.add_subplot(gs[0, j])
        ax.set_box_aspect(1)
        arrival_panel(ax, gsc, R, k, legend=(j == 0), title="eye vs. high-gain side\n" + lab)
        letter(ax, 'ABC'[j])
    ax = fig.add_subplot(gs[1, 0])
    ax.set_box_aspect(1)
    arrival_since_panel(ax, gsc, R)
    letter(ax, 'D')
    ax = fig.add_subplot(gs[1, 1:])
    ax.set_box_aspect(0.5)
    arrival_since_diff_panel(ax, gsc, R)
    letter(ax, 'E')
    save(fig, 'Figure13_arrival')


def figure13(D):
    """ The eye-position gain during compensatory head turns, far from any gaze shift, matches the gain around
    gaze shifts; the response follows where the eye is, not the direction of the last gaze shift (G, model-free);
    gaze-shift kernels don't remove it. """
    gsc = _gsc()
    T, R = gsc.pooled()
    tu = T['tuned']
    fig = plt.figure(figsize=(WIDTH, 10.2), layout='constrained')
    gs0 = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.95])
    r1 = gs0[0].subgridspec(1, 3)
    r2 = gs0[1].subgridspec(1, 3)
    r3 = gs0[2].subgridspec(1, 4)

    # A-B: example cells, all light frames
    ex = example_data(gsc)
    for j, ((sess, c), o) in enumerate(zip(EXAMPLES_13, ex)):
        ax = fig.add_subplot(r1[j])
        ax.set_box_aspect(3 / 4)
        area = np.char.decode(R[sess]['cells']['area'].astype('S'), 'utf-8')[c]
        example_panel(ax, o, area, 11 + j, j == 0)
        pk = np.nanargmax(np.nanmax(o['meas'], 0))
        print('Figure 13{}: {} cell {} ({}): at {:.0f} deg, eye temporal {:.2f} vs nasal {:.2f} (eye {:.0f} vs {:.0f} deg)'.format(
            'AB'[j], sess, c, area, (o['edges'][pk] + o['edges'][pk + 1]) / 2, o['meas'][0, pk], o['meas'][1, pk], *o['eye_means']))
        if j == 0:
            legend_above(ax, [Line2D([], [], marker='o', ls='', color=INK, ms=3.6),
                              Line2D([], [], marker='s', ls='', mfc='white', mec=INK, ms=3.6),
                              Line2D([], [], color=EYE, lw=1.6), Line2D([], [], color='#9a9a9a', lw=1.0)],
                         ['eye temporal', 'eye nasal', 'gain model', 'no gain'], 2)
        letter(ax, 'AB'[j])

    # C: the eye-position effect against time since the last gaze shift (model-free, gain cells, pillar in field)
    ax = fig.add_subplot(r1[2])
    ax.set_box_aspect(3 / 4)
    arrival_since_matched_panel(ax, gsc, R)
    letter(ax, 'C')

    # D: calibration of the predicted gain effect, gain cells
    ax = fig.add_subplot(r2[0])
    ax.set_box_aspect(1)
    lim = 0
    for k, c, lab in (('comp', EYE, 'compensatory'), ('shift', INK2, 'gaze shift'), ('still', '#b9b8b2', 'still')):
        X, Y = [], []
        for sess, r in R.items():
            if 'raw' not in r or k not in r['raw'].get('calib', {}):
                continue
            g = r['cells']['gain'].astype(bool)
            X.append(r['raw']['calib'][k]['pred'][g])
            Y.append(r['raw']['calib'][k]['meas'][g])
        X, Y = np.concatenate(X), np.concatenate(Y)
        mx, my = np.nanmean(X, 0), np.nanmean(Y, 0)
        se = np.nanstd(Y, 0) / np.sqrt(np.sum(np.isfinite(Y), 0))
        ax.errorbar(mx, my, yerr=1.96 * se, fmt='o-', color=c, ms=3.6, lw=1.2, elinewidth=0.9, capsize=0)
        lim = max(lim, np.nanmax(np.abs(np.r_[mx, my + 1.96 * se, my - 1.96 * se])))
        print('Figure 13D {}: slope of measured on predicted {:.2f}'.format(k, np.polyfit(mx, my, 1)[0]))
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.57, ls='--')
    ax.axhline(0, color=GRID, lw=0.4, zorder=0)
    ax.axvline(0, color=GRID, lw=0.4, zorder=0)
    ax.set_xlim(-1.05 * lim, 1.05 * lim)
    ax.set_ylim(-1.05 * lim, 1.05 * lim)
    ax.set_xlabel('predicted eye-gain effect\n(SD of cell noise)')
    ax.set_ylabel('measured effect')
    legend_above(ax, [Line2D([], [], color=EYE, marker='o', ms=3.6), Line2D([], [], color=INK2, marker='o', ms=3.6),
                      Line2D([], [], color='#b9b8b2', marker='o', ms=3.6)], ['compensatory', 'gaze shift', 'still'], 1)
    letter(ax, 'D')

    def beyond(key, m):
        gn, nl = T[key + '/gain'], T[key + '/null']
        m = m & np.isfinite(gn) & np.all(np.isfinite(nl), 1)
        return int(np.sum(gn[m] > np.nanmax(nl[m], 1))), int(m.sum())

    chance = 1 / (1 + N_SHUF_GSC)
    ax = fig.add_subplot(r2[1:])
    ax.set_box_aspect(3 / 8)
    hi = 0
    for i, k in enumerate(gsc.HELDOUT_CLASSES):
        kk, n = beyond('held_' + k, tu)
        hi = max(hi, _frac_bars(ax, [i], [(kk, n)], [EYE if k == 'comp' else REAL], 0.6))
        st = [r['behaviour']['state'][k] for r in R.values()]
        print('Figure 13A {}: {:.3f} of {} tuned cells predicted beyond all shuffles; {} frames; speed {:.1f} cm/s, head {:.0f} deg/s'.format(
            k, kk / max(n, 1), n, int(sum(r['heldout'][k]['n_test'] for r in R.values() if k in r['heldout'])),
            np.median([x['speed'] for x in st]), np.median([x['head_speed'] for x in st])))
    ax.axhline(chance, color=INK2, lw=0.57, ls='--')
    legend_above(ax, [Patch(color=REAL), Patch(color=EYE), Line2D([], [], color=INK2, lw=0.8, ls='--')],
                 ['gaze shifts or stillness', 'compensatory', 'chance'], 3)
    ax.set_xticks(range(len(gsc.HELDOUT_CLASSES)))
    ax.set_xticklabels(['frames with a\ngaze shift', '< 0.3 s after\na gaze shift', 'compensatory\nhead turns', 'still'])
    ax.set_xlim(-0.6, len(gsc.HELDOUT_CLASSES) - 0.4)
    ax.set_ylim(0, 1.05 * hi)
    ax.set_ylabel('frac. tuned cells predicted\nby the gain fitted elsewhere')
    letter(ax, 'E')

    ax = fig.add_subplot(r3[0])
    ax.set_box_aspect(3 / 4)
    hi = 0
    for i, a in enumerate(AREAS):
        kk, n = beyond('held_comp', tu & (T['area'] == a))
        hi = max(hi, _frac_bars(ax, [i], [(kk, n)], [AREA_COLOR[a]], 0.6))
        print('Figure 13B {}: compensatory frames, {:.3f} of {} tuned cells'.format(a, kk / max(n, 1), n))
    ax.axhline(chance, color=INK2, lw=0.57, ls='--')
    area_ticks(ax)
    ax.set_ylim(0, 1.05 * hi)
    ax.set_ylabel('frac. tuned cells predicted')
    ax.set_xlabel('compensatory head turns')
    letter(ax, 'F')

    # G: where the eye is versus how it got there (model-free, gain cells, settled frames)
    ax = fig.add_subplot(r3[1])
    ax.set_box_aspect(1)
    arrival_panel(ax, gsc, R, 'settled', legend=True)
    letter(ax, 'G')

    ax = fig.add_subplot(r3[2])
    ax.set_box_aspect(3 / 4)
    K = 'kernels/'
    hi = 0
    w = 0.26
    for i, a in enumerate(AREAS):
        m = tu & (T['area'] == a)
        hi = max(hi, _frac_bars(ax, [i - w], [(int(T[K + 'call'][m].sum()), int(m.sum()))], [AREA_COLOR[a]], w * 0.9))
        hi = max(hi, _frac_bars(ax, [i], [(int(T[K + 'call_k'][m].sum()), int(m.sum()))], [AREA_DARK[a]], w * 0.9))
        hi = max(hi, _frac_bars(ax, [i + w], [_null_frac(T[K + 'null_call_k'], m)], [SHUF], w * 0.9))
    area_ticks(ax)
    ax.set_ylim(0, 1.05 * hi)
    ax.set_ylabel('frac. tuned cells\nwith eye gain')
    legend_above(ax, [Patch(color='#9a9a9a'), Patch(color='#3a3a3a'), Patch(color=SHUF)],
                 ['without gaze-shift kernels', 'with kernels', 'with kernels, shuffled'], 1)
    letter(ax, 'H')
    print('Figure 13D: gain call {:.3f} -> {:.3f} with kernels (shuffled {:.3f})'.format(
        T[K + 'call'][tu].mean(), T[K + 'call_k'][tu].mean(), T[K + 'null_call_k'][tu].mean()))

    ax = fig.add_subplot(r3[3])
    ax.set_box_aspect(1)
    x, y = 1e3 * T[K + 'd'][tu], 1e3 * T[K + 'd_k'][tu]
    ok = np.isfinite(x) & np.isfinite(y)
    lim = np.percentile(np.abs(np.r_[x[ok], y[ok]]), 99)
    ax.scatter(np.clip(x[ok], -lim, lim), np.clip(y[ok], -lim, lim), s=2, color=INK2, lw=0, alpha=0.5, rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.57, ls='--')
    ax.axhline(0, color=GRID, lw=0.4, zorder=0)
    ax.axvline(0, color=GRID, lw=0.4, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('gain \u0394' + R2 + ' (x1e-3)\nwithout kernels')
    ax.set_ylabel('with kernels')
    letter(ax, 'I')
    print('Figure 13E: r = {:.2f}'.format(np.corrcoef(x[ok], y[ok])[0, 1]))
    save(fig, 'Figure13')


def figure14(D):
    """ Compensatory frames at matched retinal position: does a cell's response follow eye position the way its
    fitted gain says, within one compensatory stretch (no saccade between) and across a gaze shift? """
    gsc = _gsc()
    T, R = gsc.pooled()
    tu, ga = T['tuned'], T['gain']
    fig = plt.figure(figsize=(WIDTH * 0.8, 2.9), layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.1])
    min_cells = 10

    ax = fig.add_subplot(gs[0])
    ax.set_box_aspect(1)
    P = [r['behaviour']['pairs']['across'] for r in R.values() if 'sample_de' in r['behaviour']['pairs'].get('across', {})]
    de = np.concatenate([p['sample_de'] for p in P]) if P else np.zeros(0)
    dr = np.concatenate([p['sample_drho'] for p in P]) if P else np.zeros(0)
    ax.hist2d(de, dr, bins=[np.arange(-30, 31, 1.5), np.arange(-6, 6.1, 0.5)], cmap='Greys', rasterized=True)
    ax.set_xlabel('\u0394 eye in head (deg)')
    ax.set_ylabel('\u0394 retinal position (deg)')
    letter(ax, 'A')
    print('Figure 14A: compensatory pairs within one stretch: {} (too few to test; {} sessions shown); across a shift: {}'.format(
        int(sum(r['behaviour']['pairs'].get('same', {}).get('n_pairs', 0) for r in R.values())), len(P),
        int(sum(r['behaviour']['pairs'].get('across', {}).get('n_pairs', 0) for r in R.values()))))

    def paired(ax, key, col, mask, refs, lims):
        for i, a in enumerate(AREAS):
            for j, (kind, filled) in enumerate((('beh_across', True),)):
                v = T[kind + '/' + key][:, col]
                m = mask & (T['area'] == a) & np.isfinite(v)
                v = v[m]
                if len(v) < min_cells:
                    continue
                boot = np.median(np.random.default_rng(14).choice(v, (1000, len(v))), 1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                x = i
                ax.plot([x, x], np.clip([lo, hi], *lims), color=AREA_COLOR[a], lw=ERR_LW)
                ax.plot(x, np.clip(np.median(v), *lims), 'o', ms=4.5, color=AREA_COLOR[a] if filled else 'white',
                        mec=AREA_COLOR[a], mew=1.1, zorder=4)
                print('   {} {}: median {:.3f} ({:.3f} to {:.3f}), n = {}'.format(a, kind, np.median(v), lo, hi, len(v)))
        for y, c, lab in refs:
            ax.axhline(y, color=c, lw=0.85, ls='--', zorder=0)
            ax.text(1.02, y, lab, transform=ax.get_yaxis_transform(), color=c, va='center', fontsize=FONT)
        area_ticks(ax)
        ax.set_xlim(-0.6, len(AREAS) - 0.4)
        ax.set_ylim(*lims)

    ax = fig.add_subplot(gs[1])
    ax.set_box_aspect(4 / 3)
    print('Figure 14B gain expression (gain cells):')
    paired(ax, 'expr_real', 0, ga, [(0, INK2, 'none'), (1, EYE, 'as fitted')], (-1.5, 2.0))
    ax.set_ylabel('eye gain expressed\n(1 = as fitted)')
    legend_above(ax, [Line2D([], [], marker='o', ls='', color=INK2, ms=4.5)],
                 ['compensatory frames,\nacross a gaze shift'], 1)
    letter(ax, 'B')

    ax = fig.add_subplot(gs[2])
    ax.set_box_aspect(4 / 3)
    hb = np.nanmedian(T['beh_across/abc_head_model'][tu, 1])
    print('Figure 14C shift coefficient (tuned cells), head-model reference {:.2f}:'.format(hb))
    paired(ax, 'abc_real', 1, tu, [(0, EYE, 'eye'), (hb, HEAD, 'head')], (-2.0, 2.0))
    ax.set_ylabel('field shift with eye\n(0 = eye, 1 = head)')
    letter(ax, 'C')
    save(fig, 'Figure14')


def _sim_behaviour():
    """ Illustrative 40 Hz traces: still; a head-led gaze shift; a slow head turn the eye fully compensates
    (no saccade); two more gaze shifts; still. """
    dt = 0.025
    t = np.arange(0, 7.0, dt)
    hv = (120 * np.exp(-0.5 * ((t - 1.05) / 0.12) ** 2)          # fast turn carrying a gaze shift
          + 70 * np.exp(-0.5 * ((t - 3.0) / 0.14) ** 2)          # slow turn, compensated (no saccade)
          - 120 * np.exp(-0.5 * ((t - 4.95) / 0.15) ** 2))       # fast turn back, two gaze shifts
    sac_on = (0.95, 4.85, 5.15)
    sac_amp = (22.0, -14.0, -12.0)
    eye = np.zeros_like(t)
    sac = np.zeros(len(t), bool)
    e = 7.0
    for k in range(len(t)):
        for t0, a in zip(sac_on, sac_amp):
            if t0 <= t[k] < t0 + 0.075:
                e += a / 3
                sac[k] = True
                break
        else:
            e += -0.9 * hv[k] * dt
        eye[k] = e
    head = np.cumsum(hv) * dt
    return t, head, eye, head + eye, hv, sac


def fig13_schematic(D):
    """ How Figures 13 and 14 compare compensatory head turns with the rest of the recording. """
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(WIDTH, 7.0), layout='constrained')
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 2.6])
    top = gs[0].subgridspec(2, 1, height_ratios=[3, 0.55], hspace=0.04)
    bot = gs[1].subgridspec(1, 2, width_ratios=[1.35, 1])

    # A: label every imaging frame by behaviour
    t, head, eye, gaze, hv, sac = _sim_behaviour()
    ax = fig.add_subplot(top[0])
    ax.plot(t, head - head[0], color=HEAD, lw=1.4)
    ax.plot(t, eye, color=EYE, lw=1.4)
    ax.plot(t, gaze - gaze[0], color=INK, lw=1.4)
    ax.set_xlim(0, t[-1])
    ax.set_ylabel('angle (deg)')
    ax.tick_params(labelbottom=False)
    legend_above(ax, [Line2D([], [], color=HEAD, lw=1.4), Line2D([], [], color=EYE, lw=1.4), Line2D([], [], color=INK, lw=1.4)],
                 ['head', 'eye in head', 'gaze = head + eye'], 3)
    letter(ax, 'A')
    ax2 = fig.add_subplot(top[1], sharex=ax)
    gv = np.gradient(gaze, t)
    on = t[np.flatnonzero(np.diff(np.r_[0, sac.astype(int)]) == 1)]
    end = t[np.flatnonzero(np.diff(np.r_[sac.astype(int), 0]) == -1)]
    fr = np.arange(0, t[-1], 2 / 15)
    cls_col = {'shift': SHIFT_C, 'post': EARLY_C, 'comp': EYE, 'still': '#ffffff', 'other': '#efeeea'}
    for f0 in fr:
        m = (t >= f0) & (t < f0 + 2 / 15)
        inside = np.any(sac[m])
        last = end[end <= f0]
        since = f0 - last[-1] if len(last) else np.inf
        nxt = on[on >= f0 + 2 / 15]
        until = nxt[0] - (f0 + 2 / 15) if len(nxt) else np.inf
        h, g = np.mean(hv[m]), np.mean(gv[m])
        c = ('shift' if inside else 'post' if since < 0.3 else
             'comp' if (abs(h) > 60 and abs(g) < 120 and until >= 0.1) else 'still' if abs(h) < 10 else 'other')
        ax2.add_patch(plt.Rectangle((f0, 0), 2 / 15, 1, facecolor=cls_col[c], edgecolor=INK2, lw=0.3))
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_ylabel('frames', rotation=0, ha='right', va='center')
    ax2.set_xlabel('time (s); illustrative traces, one box per 133 ms imaging frame')
    for sp in ('left', 'top', 'right'):
        ax2.spines[sp].set_visible(False)
    leg = ax2.legend([Patch(facecolor=cls_col[k], edgecolor=INK2, lw=0.3) for k in ('shift', 'post', 'comp', 'still', 'other')],
                     ['gaze shift', '< 0.3 s after a shift', 'compensatory: head turning, gaze steady, >= 0.3 s after / 0.1 s before a shift',
                      'still', 'other'], ncol=3, loc='upper left', bbox_to_anchor=(0, -1.6), borderaxespad=0)

    # B: held-out prediction (Figure 13A-B)
    ax = fig.add_subplot(bot[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.4, 10)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#f4f3ef', ec=INK2, fs=FONT):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15,rounding_size=0.25', fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9, mutation_scale=9))

    strip_y = 8.6
    for i in range(30):
        c = EYE if i in (6, 7, 15, 16, 17, 24) else '#cfcec9'
        ax.add_patch(plt.Rectangle((0.2 + i * 0.32, strip_y), 0.3, 0.7, fc=c, ec='white', lw=0.5))
    ax.text(0.2, strip_y + 0.95, 'all light frames of a session', fontsize=FONT, color=INK2, va='bottom')
    box(0.2, 5.3, 4.4, 2.2, 'fit ADD and GAIN\non every OTHER frame\n(+ 10 GAIN fits with\neye position shuffled)')
    box(5.4, 5.3, 4.4, 2.2, 'predict the\nheld-out\ncompensatory frames', fc='#d6f3f6', ec=EYE)
    arrow(2.4, strip_y - 0.1, 2.4, 7.7)
    arrow(7.6, strip_y - 0.1, 7.6, 7.7)
    arrow(4.75, 6.4, 5.25, 6.4)
    box(0.2, 2.3, 9.6, 2.2, 'on the held-out frames:  \u0394' + R2 + ' = ' + R2 + '(GAIN) \u2212 ' + R2 + '(ADD)\n'
        'a cell counts if its real \u0394' + R2 + ' beats\nall 10 shuffled fits (chance 1 in 11)')
    arrow(7.6, 5.15, 7.6, 4.65)
    box(0.2, 0.0, 9.6, 1.6, 'repeated with each class held out in turn:\ngaze shift, < 0.3 s after, compensatory, still')
    arrow(5.0, 2.15, 5.0, 1.75)
    letter(ax, 'B', y=10)

    # C: matched pairs (Figure 14)
    ax = fig.add_subplot(bot[1])
    ax.set_box_aspect(0.95)
    r = np.linspace(-180, 180, 361)
    f = np.exp(2.0 * (np.cos(np.radians(r + 20)) - 1))
    for gsc_, ls, lab in ((0.75, LS_LOW, 'eye left'), (1.25, LS_HIGH, 'eye right')):
        ax.plot(r, gsc_ * f, color=INK, ls=ls, lw=1.4, label=lab)
    r0 = -40
    y0 = np.exp(2.0 * (np.cos(np.radians(r0 + 20)) - 1))
    ax.plot([r0, r0], [0.75 * y0, 1.25 * y0], color=EYE, lw=2.2)
    ax.plot([r0, r0], [0.75 * y0, 1.25 * y0], 'o', color=EYE, ms=5)
    ax.annotate('two compensatory frames\nfrom different fixations:\nsame retinal position (< 5 deg),\neye 3-30 deg apart\n\n'
                'regress \u0394 response on the\n\u0394 the fitted gain term predicts:\nslope 1 = as fitted, 0 = none',
                xy=(r0 + 3, 1.0 * y0), xytext=(48, 0.42), fontsize=FONT, color=INK,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel('retinal position of the pillar (deg)')
    ax.set_ylabel('response')
    ax.set_yticks([])
    legend_above(ax, [Line2D([], [], color=INK, ls=LS_LOW, lw=1.4), Line2D([], [], color=INK, ls=LS_HIGH, lw=1.4)],
                 ['eye left', 'eye right'], 2)
    letter(ax, 'C')
    save(fig, 'fig13_shematic')


FIGURES = {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5, 6: figure6, 7: figure7, 8: figure8,
           9: figure9, 10: figure10, 11: figure11, '10cells': figure10_cells, '11cells': figure11_cells,
           12: figure12, 13: figure13, '13arrival': figure13_arrival, 14: figure14,
           'fig13_schematic': fig13_schematic}

if __name__ == '__main__':
    data = load()
    which = [int(a) if a.isdigit() else a for a in sys.argv[1:]] or list(FIGURES)
    for n in which:
        FIGURES[n](data)
