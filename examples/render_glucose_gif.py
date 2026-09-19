"""Render a three-model, five-scenario GIF gallery from saved traces.

python examples/render_glucose_gif.py --trace-dir /path/to/scenarios --output-dir /new/gallery
Requires matplotlib, NumPy, Pillow. Reads the calibration check's summary.json
and NPZ files; verifies hashes before rendering. Does not run simulations.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MODELS = [('t1d_balance', 't1d', 'T1D', '#087e8b'),
          ('t2d_insulin', 't2d', 'T2D', '#a05d16'),
          ('no_pump_insulin', 't2d_no_pump', 'T2D no-pump', '#7954a3')]
CASES = [('fasting', 'Fasting'), ('meal', 'Meal only'),
         ('bolus', 'Bolus only'), ('meal_bolus', 'Meal + bolus'),
         ('exercise', 'Exercise')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.trace_dir / 'summary.json').read_text())
    traces, sensors, hashes = {}, {}, {}
    for config, kind, _, _ in MODELS:
        for case, _ in CASES:
            row, = [r for r in summary['rows'] if r['config'] == config and r['pattern'] == case]
            path = args.trace_dir / row['artifact']
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != row['artifact_sha256']:
                raise ValueError(f'Trace hash mismatch: {path}')
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                glucose = data['glucose'].copy()
                cgm = data['cgm'].copy() if 'cgm' in data else None
            if glucose.shape != (181,) or not np.isfinite(glucose).all():
                raise ValueError(f'Invalid trace: {path}')
            if cgm is not None:
                if cgm.shape != glucose.shape or np.isinf(cgm).any() or not np.isfinite(cgm).any():
                    raise ValueError(f'Invalid CGM trace: {path}')
                sensors[kind, case] = cgm
            traces[kind, case] = glucose
            hashes[path.name] = digest
    # Use identical axes across the entire gallery; never clip any trace.
    all_traces = list(traces.values()) + list(sensors.values())
    upper = max(250, np.ceil(max(np.nanmax(g) for g in all_traces) / 50) * 50)
    lower = min(0, np.floor(min(np.nanmin(g) for g in all_traces) / 50) * 50)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for config, kind, title, color in MODELS:
        for case, label in CASES:
            glucose = traces[kind, case]
            fig, ax = plt.subplots(figsize=(4.8, 3.5), facecolor='#f5f8fc')
            ax.axhspan(70, 180, color='#e6f4ea')
            for threshold in (70, 180):
                ax.axhline(threshold, color='#ad4554', lw=.8, ls='--')
            if case in ('meal', 'meal_bolus'):
                ax.axvspan(30, 45, color='#db7b17', alpha=.15)
            if case in ('bolus', 'meal_bolus'):
                ax.axvline(30, color='#7954a3', lw=1, ls=':')
            if case == 'exercise':
                ax.axvspan(60, 90, color='#3183bb', alpha=.15)
            line, = ax.plot([], [], color=color, lw=2, label='Plasma')
            sensor, = ax.plot([], [], color='#52657d', lw=.8, alpha=.8, label='CGM')
            if (kind, case) in sensors:
                ax.legend(loc='lower right', fontsize=8)
            ax.set(xlim=(0, 180), ylim=(lower, upper), xlabel='Minutes',
                   ylabel='Glucose (mg/dL)', title=f'{title} · {label}')
            ax.set_xticks([0, 60, 120, 180])
            ax.spines[['top', 'right']].set_visible(False)
            clock = ax.text(.97, .93, '', ha='right', transform=ax.transAxes, fontsize=9)
            caption = ('Default noise + circadian · separate CGM readout'
                       if (kind, case) in sensors else 'Saved direct-ODE trace · noise off')
            fig.text(.5, .02, caption, ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .04, 1, 1))
            frames = []
            for index in range(0, 181, 5):
                line.set_data(np.arange(index + 1), glucose[:index + 1])
                if (kind, case) in sensors:
                    sensor.set_data(np.arange(index + 1), sensors[kind, case][:index + 1])
                clock.set_text(f'{index} / 180 min')
                fig.canvas.draw()
                frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
            target = args.output_dir / f'{kind}_{case}.gif'
            frames[0].save(target, save_all=True, append_images=frames[1:], loop=0,
                           duration=[150] * (len(frames) - 1) + [1500],
                           comment=json.dumps({'trace_sha256': hashes[f'{config}_{case}.npz']}).encode())
            plt.close(fig)
            print(target, flush=True)
    print(json.dumps({'inputs': hashes}, indent=2))


if __name__ == '__main__':
    main()
