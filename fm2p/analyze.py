# -*- coding: utf-8 -*-
"""
fm2p/analyze.py

Post-preprocessing analysis chain for freely moving recordings, one call per
session or per batch.

DMM, October 2026
"""


if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import os
import sys
import time
import argparse
import subprocess
import numpy as np

from .utils.paths import find


DEFAULT_BATCH_PATH = '/home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings'

STEPS = ('pose', 'retinocentric', 'fixed_pupil', 'gaze_stab', 'behavior', 'nonvisual')

RENDER_FLAGS = ['--walls', '--invert_pitch', '--eye_smooth_Q', '-1']


def find_recordings(root):

    files = find('*_preproc.h5', root, MR=False, retempty=True) or []
    by_dir = {}
    for p in files:
        d = os.path.dirname(p)
        if 'boundary' in os.path.basename(p) or not os.path.basename(d).startswith('fm'):
            continue
        if d not in by_dir or os.path.getmtime(p) > os.path.getmtime(by_dir[d]):
            by_dir[d] = p
    return [by_dir[d] for d in sorted(by_dir)]


def step_outputs(h5_path):

    pre = h5_path.replace('_preproc.h5', '')
    return {
        'pose': pre + '_retinal_pose.npz',
        'retinocentric': pre + '_retinal_images_retinocentric.npz',
        'fixed_pupil': pre + '_retinal_images_fixed_pupil_ego.npz',
        'gaze_stab': pre + '_gaze_stab.h5',
        'behavior': pre + '_neural.h5',
        'nonvisual': pre + '_nonvisual.h5',
    }


def _is_current(out, inputs, npz_keys=()):

    if not os.path.isfile(out):
        return False
    t = os.path.getmtime(out)
    if any(os.path.isfile(i) and os.path.getmtime(i) > t for i in inputs):
        return False
    if npz_keys:
        try:
            with np.load(out, allow_pickle=False) as z:
                return all(k in z.files for k in npz_keys)
        except Exception:
            return False
    return True


def _run(cmd, log_path, env=None):

    with open(log_path, 'a') as log:
        log.write('\n$ {}\n'.format(' '.join(cmd)))
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
        log.write('[exit {}]\n'.format(proc.returncode))
    return proc.returncode


def _step_command(step, h5_path, out, batch_mode, exists=os.path.isfile):

    py = [sys.executable, '-u', '-m']

    gui = ['--no_arena_gui'] if batch_mode else []
    if step == 'pose':
        return py + ['fm2p.get_retinal_image', '--h5_path', h5_path] + RENDER_FLAGS + gui + \
            ['--pose_only', '--out_npz', out['pose']], ''
    if step == 'retinocentric':
        return py + ['fm2p.get_retinal_image', '--h5_path', h5_path] + RENDER_FLAGS + gui + \
            ['--out_npz', out['retinocentric']], ''
    if step == 'fixed_pupil':
        return py + ['fm2p.get_retinal_image', '--h5_path', h5_path] + RENDER_FLAGS + gui + \
            ['--fixed_pupil', '--out_npz', out['fixed_pupil']], ''
    if step == 'gaze_stab':
        if not exists(out['retinocentric']):
            return None, os.path.basename(out['retinocentric'])
        return py + ['fm2p.gaze_stabilization', '--h5_path', h5_path,
                     '--npz_path', out['retinocentric']], ''
    if step == 'behavior':
        if not exists(out['pose']):
            return None, os.path.basename(out['pose'])
        cmd = py + ['fm2p.new_analysis', '--h5_path', h5_path, '--npz_path', out['pose']]

        if exists(out['retinocentric']):
            cmd += ['--images_npz', out['retinocentric']]
        return cmd, ''
    if step == 'nonvisual':
        if not exists(out['pose']):
            return None, os.path.basename(out['pose'])
        return py + ['fm2p.nonvisual_probe', '--h5_path', h5_path, '--npz_path', out['pose']], ''
    raise ValueError('Unknown step {}'.format(step))


def analyze_recording(h5_path, steps=STEPS, skip_existing=False, batch_mode=False, dry_run=False):

    out = step_outputs(h5_path)
    log_path = h5_path.replace('_preproc.h5', '_analyze.log')

    inputs = {'pose': [h5_path], 'retinocentric': [h5_path], 'fixed_pupil': [h5_path],
              'gaze_stab': [h5_path, out['retinocentric']],
              'behavior': [h5_path, out['pose'], out['retinocentric']],
              'nonvisual': [h5_path, out['pose']]}
    needs = {'gaze_stab': 'retinocentric', 'behavior': 'pose', 'nonvisual': 'pose'}
    render_keys = ('phi_rest_ref',)

    env = dict(os.environ)
    if batch_mode:
        env['MPLBACKEND'] = 'Agg'

    if not dry_run:
        with open(log_path, 'a') as log:
            log.write('\n===== analyze {} =====\n'.format(time.strftime('%Y-%m-%d %H:%M:%S')))

    status = {}

    failed, planned = set(), set()

    def exists(path):

        return path not in failed and (os.path.isfile(path) or path in planned)

    for step in steps:
        dep = needs.get(step)
        if dep is not None and (status.get(dep, '').startswith('failed') or out[dep] in failed):
            status[step] = 'skipped ({} failed)'.format(dep)
            print('  [{}] skipped: {} failed.'.format(step, dep))
            continue
        keys = render_keys if step in ('pose', 'retinocentric', 'fixed_pupil') else ()
        if skip_existing and _is_current(out[step], inputs[step], keys):
            status[step] = 'skipped (current)'
            print('  [{}] skipped: {} is current.'.format(step, os.path.basename(out[step])))
            continue
        cmd, missing = _step_command(step, h5_path, out, batch_mode, exists=exists)
        if cmd is None:
            status[step] = 'skipped (no {})'.format(missing)
            print('  [{}] skipped: no {}.'.format(step, missing))
            continue
        if dry_run:
            print('  [{}] {}'.format(step, ' '.join(cmd[1:])))
            status[step] = 'dry run'
            planned.add(out[step])
            continue
        if dep is not None:
            with np.load(out[dep], allow_pickle=False) as z:
                if 'phi_rest_ref' not in z.files:
                    print('  WARNING: {} predates the vertical eye rest reference -- '
                          're-run the {} step.'.format(os.path.basename(out[dep]), dep))
        print('\n--- [{}] {} ---'.format(step, os.path.basename(h5_path)))
        t0 = time.time()
        code = _run(cmd, log_path, env=env)
        dt = time.time() - t0
        status[step] = 'ok' if code == 0 else 'failed (exit {})'.format(code)
        if code != 0:
            failed.add(out[step])
        print('--- [{}] {} in {:.0f} s ---'.format(step, status[step], dt))
    return status


def analyze_batch(batch_path=DEFAULT_BATCH_PATH, start_at=1, steps=STEPS, skip_existing=False,
                  pool=False, dry_run=False, batch_mode=True):

    steps = [s for s in STEPS if s in steps]
    recs = find_recordings(batch_path)
    n = len(recs)
    print('Found {} fm recording(s) with a preproc.h5 under {}.'.format(n, batch_path))
    print('Steps: {}{}.'.format(', '.join(steps), ' (skipping current outputs)' if skip_existing else ''))
    if start_at > 1:
        print('Resuming at recording {}/{}.'.format(start_at, n))

    results = {}
    t_all = time.time()
    for i in range(start_at - 1, n):
        h5 = recs[i]
        print('\n=== [{}/{}] {} ==='.format(i + 1, n, h5))
        try:
            results[h5] = analyze_recording(h5, steps, skip_existing=skip_existing,
                                            batch_mode=batch_mode, dry_run=dry_run)
        except Exception as e:
            print('  ERROR in {}: {}'.format(h5, e))
            results[h5] = {'error': str(e)}

    if pool and not dry_run:
        neural = [step_outputs(h)['behavior'] for h in recs if os.path.isfile(step_outputs(h)['behavior'])]
        if neural:
            print('\n=== Pooling {} _neural.h5 file(s) by area ==='.format(len(neural)))
            log = os.path.join(batch_path, 'analyze_pool.log')
            _run([sys.executable, '-u', '-m', 'fm2p.new_analysis', '--pool'] + neural, log)

    print('\nAnalysis complete in {:.1f} min.  Per recording ({}):'.format(
        (time.time() - t_all) / 60, ', '.join(steps)))
    n_bad = 0
    for h5, st in results.items():
        bad = [k for k, v in st.items() if v.startswith('failed') or k == 'error'
               or v.endswith('failed)')]
        n_bad += bool(bad)
        print('  {} {}'.format('FAIL' if bad else 'ok  ', os.path.basename(h5).replace('_preproc.h5', '')))
        if bad:
            for k in bad:
                print('       {}: {}'.format(k, st[k]))
    print('{} of {} recording(s) had a failed step.  Logs: <prefix>_analyze.log beside each '
          'preproc.h5.'.format(n_bad, len(results)))
    return results


def main():

    ap = argparse.ArgumentParser(description='Post-preprocessing analysis chain for fm recordings.')
    ap.add_argument('-d', '--dir', type=str, default=None,
                    help='Session directory; runs every fm recording in it.')
    ap.add_argument('--batch', action='store_true', default=False,
                    help='Run every fm recording found under --batch_path.')
    ap.add_argument('--batch_path', type=str, default=DEFAULT_BATCH_PATH,
                    help='Root directory to search when --batch is set (default: %(default)s).')
    ap.add_argument('--batch_start', type=int, default=1,
                    help='1-indexed recording to resume --batch from (same order as '
                         'preprocess --batch).  Default: 1.')
    ap.add_argument('--steps', nargs='+', choices=STEPS, default=list(STEPS),
                    help='Steps to run, always in pipeline order (default: all).')
    ap.add_argument('--skip_existing', action='store_true',
                    help='Skip a step whose output is newer than its inputs (renders must '
                         'also carry the vertical eye rest reference).')
    ap.add_argument('--pool', action='store_true',
                    help='After the batch, pool every _neural.h5 by area (new_analysis --pool).')
    ap.add_argument('--dry_run', action='store_true',
                    help='Print the commands without running anything.')
    args = ap.parse_args()

    if args.batch:
        analyze_batch(args.batch_path, start_at=args.batch_start, steps=args.steps,
                      skip_existing=args.skip_existing, pool=args.pool, dry_run=args.dry_run)
    else:
        if args.dir is None:
            ap.error('-d/--dir is required unless --batch is given.')
        analyze_batch(args.dir, start_at=args.batch_start, steps=args.steps,
                      skip_existing=args.skip_existing, pool=args.pool, dry_run=args.dry_run,
                      batch_mode=False)


if __name__ == '__main__':
    main()


# python -m fm2p.analyze --batch --batch_path /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings
