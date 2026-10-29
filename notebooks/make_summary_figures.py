import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
    fig = plt.figure(figsize=(WIDTH, 3.95), layout='constrained')
    gs = fig.add_gridspec(1, 2)

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
    legend_above(ax, [Patch(color=INK2), Patch(color=SHUF)], ['real', 'head-on-body shuffled'], 2, y=1.1)
    letter(ax, 'B')
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
                if h == 0 and j == 0 and i == 0:
                    legend_above(ax, [Line2D([], [], color=INK, lw=1.59, ls=LS_LOW), Line2D([], [], color=INK, lw=1.59, ls=LS_HIGH)],
                                 ['eye right', 'eye left'], 2)
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


FIGURES = {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5, 6: figure6, 7: figure7, 8: figure8}

if __name__ == '__main__':
    data = load()
    which = [int(a) for a in sys.argv[1:]] or sorted(FIGURES)
    for n in which:
        FIGURES[n](data)
