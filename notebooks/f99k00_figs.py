
import contextlib
import io
import os
import pickle
import sys

import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import make_summary_figures as S
import make_summary_figures_v02 as V
import eye_vs_gaze_tests as G

plt = V.plt
OUT = os.path.join(HERE, 'f99k00_figures')
plt.rcParams['svg.fonttype'] = 'none'

TRACE_SESSION = '251025_DMM_DMM056_pos18'
TRACE_T0, TRACE_S = 2855.33, 0.84
TRACE_CACHE = os.path.join(HERE, 'manuscript_figures', 'f99k00_trace.npz')
SHIFT_VEL = 240.0
GAZE_PKL = '/home/dylan/Storage/freely_moving_data/V1PPC/cross_area_cache/gazeweight_all.pkl'


def save(fig, name):
    import re
    W, H = fig.get_size_inches()
    if H > V.MAX_HEIGHT + 1e-6:
        print('  WARNING: {} is {:.2f} in tall on the page (limit {:.1f})'.format(name, H * V.SCALE, V.OUT_MAX_HEIGHT))
    bw = V.bar_widths(fig)
    if bw:
        print('  bar widths on the page: {:.3f}-{:.3f} in (target {:.3f})'.format(min(bw.values()), max(bw.values()),
                                                                                 V.BAR * V.SCALE))
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, name)
    fig.savefig(base + '.png', dpi=V.DPI * V.SCALE, pil_kwargs={'dpi': (V.DPI, V.DPI)})
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    svg = re.sub(r'width="[\d.]+pt" height="[\d.]+pt"',
                 'width="{:.2f}pt" height="{:.2f}pt"'.format(72 * W * V.SCALE, 72 * H * V.SCALE), svg, count=1)
    with open(base + '.svg', 'w') as f:
        f.write(svg)
    plt.close(fig)
    print('saved {}.png/.svg ({:.2f} x {:.2f} in)'.format(base, W * V.SCALE, H * V.SCALE))

def trace_window():
    if os.path.isfile(TRACE_CACHE):
        z = np.load(TRACE_CACHE)
        if str(z['session']) == TRACE_SESSION and float(z['t0']) == TRACE_T0 and float(z['dur']) == TRACE_S:
            return {k: z[k] for k in ('t', 'eye', 'head', 'gaze')}
    gsc = S._gsc()
    neural = [p for p, _, sess, _ in gsc.sessions() if sess == TRACE_SESSION][0]
    na = gsc.na
    fs0, dt0 = na.FS, na.DT
    na.FS, na.DT = 60.0, 1.0 / 60.0
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            s = na.load_session(neural.replace('_neural.h5', '_preproc.h5'),
                                neural.replace('_neural.h5', '_retinal_pose.npz'))
    finally:
        na.FS, na.DT = fs0, dt0
    m = (s['t'] >= TRACE_T0) & (s['t'] < TRACE_T0 + TRACE_S)
    head = s['yaw_r'][m]
    eye = na._wrap(-(s['pfh'][m] - s['mean_pfh']))
    out = {'t': s['t'][m] - TRACE_T0, 'eye': eye, 'head': head, 'gaze': head + eye}
    if not (np.all(np.isfinite(eye)) and np.all(np.isfinite(head)) and s['light'][m].all()):
        print('WARNING: example window has missing samples or dark frames')
    os.makedirs(os.path.dirname(TRACE_CACHE), exist_ok=True)
    np.savez(TRACE_CACHE, session=TRACE_SESSION, t0=TRACE_T0, dur=TRACE_S, **out)
    return out


def shift_spans(t, gaze):
    v = np.abs(np.gradient(gaze, t))
    d = np.diff(np.r_[0, (v > SHIFT_VEL).astype(int), 0])
    dt = np.median(np.diff(t))
    return [(t[a] - dt / 2, t[b - 1] + dt / 2) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]

def trace_panels(fig, cell):
    z = trace_window()
    t = z['t']
    spans = shift_spans(t, z['gaze'])
    h0 = z['head'][0]
    rows = (('eye', z['eye'], 'eye in\nhead (deg)'), ('head', z['head'] - h0, 'head\n(deg)'),
            ('gaze', z['gaze'] - h0, 'gaze\n(deg)'))
    sub = cell.subgridspec(3, 1, height_ratios=[1, 1.45, 1.45])
    axs = []
    for i, (k, y, lab) in enumerate(rows):
        ax = fig.add_subplot(sub[i], sharex=axs[0] if axs else None)
        for a, b in spans:
            ax.axvspan(a, b, color=V.GRID, lw=0, zorder=0)
        ax.plot(t, y, color=V.INK, lw=0.9)
        ax.set_ylabel(lab)
        ax.set_xlim(0, TRACE_S)
        if i < 2:
            ax.tick_params(labelbottom=False)
        axs.append(ax)
    lo = min(np.min(r[1]) for r in rows[1:])
    hi = max(np.max(r[1]) for r in rows[1:])
    pad = 0.06 * (hi - lo)
    for ax in axs[1:]:
        ax.set_ylim(lo - pad, hi + pad)
    axs[-1].set_xlabel('time (s)')
    axs[-1].set_xticks(np.arange(0, TRACE_S + 1e-9, 0.2))
    print('Trace: {} {:.1f}-{:.1f} s, {} gaze shifts, head turn {:.0f} deg, eye range {:.0f} deg'.format(
        TRACE_SESSION, TRACE_T0, TRACE_T0 + TRACE_S, len(spans), z['head'][-1] - h0, np.ptp(z['eye'])))
    return axs


def gaze_weight_panel(ax):
    with open(GAZE_PKL, 'rb') as f:
        P = pickle.load(f)
    rows, T = G.cell_table(P['res'], P['area'])
    names = list(V.AREAS)
    est, lo, hi = [], [], []
    for a in names:
        idx = np.flatnonzero(rows['area'] == a)
        est.append(G.peak(G.SPLIT_W, G.split_profile(T, 'real', idx, rows['de'])))
        b = G.boot(lambda k: G.peak(G.SPLIT_W, G.split_profile(T, 'real', idx[k], rows['de'])), len(idx))
        l, h = np.nanpercentile(b, [2.5, 97.5])
        lo.append(l), hi.append(h)
        print('Gaze weight {}: {:.2f} [{:.2f}, {:.2f}], {} cells'.format(a, est[-1], l, h, len(idx)))
    est, lo, hi = map(np.array, (est, lo, hi))
    x = np.arange(len(names))
    for i, a in enumerate(names):
        ax.errorbar(x[i], est[i], yerr=[[est[i] - lo[i]], [hi[i] - est[i]]], color=V.AREA_COLOR[a], marker='o', ms=3.6,
                    ls='none', elinewidth=0.9, capsize=0)
    ax.axhline(0, color=V.INK2, lw=0.57, ls='--')
    ax.axhline(1, color=V.INK2, lw=0.57, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlim(-0.4, len(names) - 0.6)
    ax.set_ylim(min(lo.min(), 0) * 1.1 - 0.05, 1.08 * hi.max())
    ax.set_ylabel('heading-gain shift /\neye movement')
    V.legend_above(ax, [Line2D([], [], color=V.INK2, lw=0.8, ls='--'), Line2D([], [], color=V.INK2, lw=0.8, ls=':')],
                   ['eye-in-head', 'gaze-in-world'], 1)

F1_HEIGHT = 7.2
F1_COLS = [2.35, 1.9, 1.9, 2.35]
TEMPORAL_NASAL = ('eye temporal (low third)', 'eye nasal (high third)')


def figure1():
    d = V.load()['fig1']
    gsc = S._gsc()
    _, R = gsc.pooled()
    fig = plt.figure(figsize=(V.WIDTH, F1_HEIGHT), layout='constrained')
    gs = fig.add_gridspec(1, 4, width_ratios=F1_COLS)
    c1 = gs[0].subgridspec(3, 1)
    c4 = gs[3].subgridspec(3, 1)

    axs = trace_panels(fig, c1[0:2, 0])
    V.letter(axs[0], 'A')
    ax = fig.add_subplot(c1[2])
    ax.set_box_aspect(3 / 4)
    V.eye_position_hist(ax, d)
    ax.set_xticks([-30, 0, 30])
    V.letter(ax, 'B')

    before = set(fig.axes)
    cg = gs[1:3].subgridspec(4, 2)
    V.example_fields(fig, [cg[j, 0] for j in range(4)], [cg[j, 1] for j in range(4)], d)
    new = [a for a in fig.axes if a not in before]
    for j in range(4):
        for k, ax in enumerate(new[2 * j:2 * j + 2]):
            ax.set_title('')
            ax.set_box_aspect(1)
            ax.set_xticks([-180, 0, 180])
            ax.set_ylabel('cell {}\nrate'.format(j + 1) if k == 0 else '')
            if j < 3:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(('retinal\nazimuth (deg)', 'head-centered\nazimuth (deg)')[k])
    axC = new[0]
    leg = axC.get_legend()
    handles = leg.legend_handles
    leg.remove()
    V.legend_above(axC, handles, list(TEMPORAL_NASAL), 1)
    V.letter(axC, 'C')

    ax = fig.add_subplot(c4[0])
    ax.set_box_aspect(3 / 4)
    S.arrival_since_matched_panel(ax, gsc, R)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlabel('time since last\ngaze shift (s)')
    leg = ax.get_legend()
    handles, labels = leg.legend_handles, [t.get_text() for t in leg.texts]
    leg.remove()
    V.legend_above(ax, handles, labels, 1)
    V.letter(ax, 'D')
    ax = fig.add_subplot(c4[1])
    ax.set_box_aspect(3 / 4)
    S.arrival_panel(ax, gsc, R, 'settled', legend=True)
    V.letter(ax, 'E')
    ax = fig.add_subplot(c4[2])
    ax.set_box_aspect(3 / 4)
    gaze_weight_panel(ax)
    V.letter(ax, 'F')
    save(fig, 'Figure1')


F2_TOP = [3.56, 2.72, 2.72]
F2_BOTTOM = [1.5, 1.5, 1.5, 1.5, 0.4]


def frame_calls(ax, d):
    cols = [('gm_eye_better', V.EYE), ('gm_eye_better_dark', V.EYE_DARK), ('gm_head_better', V.HEAD),
            ('gm_head_better_dark', V.HEAD_DARK)]
    w = 0.2
    for i, a in enumerate(V.AREAS):
        for j, (col, color) in enumerate(cols):
            v = d['G'][a][col]
            x = i + (j - 1.5) * w
            ax.bar(x, v['frac'], width=w * 0.9, color=color)
            V.errbar(ax, x, v['lo'], v['hi'])
    V.area_ticks(ax)
    ax.set_ylabel('frac. cells')
    ax.set_ylim(0, 1.05 * max(d['G'][a][c]['hi'] for a in V.AREAS for c, _ in cols))
    return V.legend_above(ax, [V.Patch(color=c) for _, c in cols],
                          ['eye (light)', 'eye (dark)', 'head (light)', 'head (dark)'], 2)


def figure2():
    D = V.load()
    fig = plt.figure(figsize=(V.WIDTH, 6.), layout='constrained')
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 3.2])
    top = gs[0].subgridspec(1, 3, width_ratios=F2_TOP)
    bottom = gs[1].subgridspec(1, 5, width_ratios=F2_BOTTOM)
    title = lambda ax, t: ax.set_title(t, fontsize=V.FONT, pad=4)

    axA = fig.add_subplot(top[0])
    frame_calls(axA, D['fig1'])

    d = D['fig3']
    axB1 = fig.add_subplot(top[1])
    axB2 = fig.add_subplot(top[2], sharey=axB1)
    hi = max(V.sim_calls(axB1, d, 'A'), V.sim_calls(axB2, d, 'B'))
    axB1.set_ylim(0, 1.05 * hi)
    axB1.set_ylabel('frac. tuned cells')
    axB2.tick_params(labelleft=True)
    title(axB1, 'called eye-centered')
    title(axB2, 'called head-centered')
    legB = V.legend_above(axB1, [V.Patch(facecolor=c, hatch=V.HATCH if k != 'real' else None,
                                         edgecolor=V.INK2 if k != 'real' else 'none') for k, c in V.SIM_BARS],
                          ['real', 'simulated eye-centered + gain', 'simulated head-centered'], 1, y=1.13)
    V.letter(axA, 'A', level=legB)
    V.letter(axB1, 'B', level=legB)

    d5 = D['fig5']
    axC1, axC2 = fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1])
    V.area_pairs(axC1, lambda a, k: (d['D'][a][k]['median'], d['D'][a][k]['p10'], d['D'][a][k]['p90'], None))
    V.area_pairs(axC2, lambda a, k: (d5['B'][a][k]['median'], d5['B'][a][k]['p10'], d5['B'][a][k]['p90'], None))
    for ax_ in (axC1, axC2):
        ax_.set_ylim(0, 0.75)
    axC1.set_ylabel('per-cell signal (x1e\u22123)')
    title(axC1, 'head-centered\nreadout')
    title(axC2, 'body-centered\nreadout')
    legC = V.legend_above(axC1, [V.Patch(color=V.INK2), V.Patch(color=V.SHUF)], ['real', 'shuffled'], 1, y=1.3)
    V.letter(axC1, 'C', level=legC)

    d4 = D['fig4']
    axD, axE = fig.add_subplot(bottom[2]), fig.add_subplot(bottom[3])
    V.vis_never_bars(axD, d4, 'need')
    axD.set_ylabel('gain signal (x100)')
    V.vis_never_bars(axE, d4, 'coverage')
    axE.set_ylabel('coverage')
    V.legend_above(axD, [V.Patch(color='#9a9a9a'), V.Patch(color='#3a3a3a')], ['visible', 'never visible'], 1, y=1.3)
    V.letter(axD, 'D', level=legC)
    V.letter(axE, 'E', level=legC)
    save(fig, 'Figure2')


FIGURES = {1: figure1, 2: figure2}

if __name__ == '__main__':
    which = [int(a) for a in sys.argv[1:]] or list(FIGURES)
    for n in which:
        FIGURES[n]()
