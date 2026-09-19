"""Render README schematics without running simulations.

Optional documentation dependency: python -m pip install Pillow
Run from the repository root: python examples/render_readme_assets.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from matplotlib import get_data_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'assets'
BG, INK, MUTED, TEAL = '#f5f8fc', '#172b45', '#52657d', '#087e8b'


def font(size):
    return ImageFont.truetype(str(Path(get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'), size)


def canvas(title, subtitle, height):
    im = Image.new('RGB', (1200, height), BG)
    d = ImageDraw.Draw(im)
    d.text((42, 26), title, font=font(32), fill=INK)
    d.text((42, 74), subtitle, font=font(18), fill=MUTED)
    return im, d


def card(d, box, title, lines, active=False):
    x, y, _, _ = box
    d.rounded_rectangle(box, radius=16, fill='#e0f2f3' if active else 'white',
                        outline=TEAL if active else '#c9d5e2', width=3 if active else 2)
    d.text((x + 20, y + 20), title, font=font(23), fill=INK)
    for i, line in enumerate(lines):
        d.text((x + 20, y + 65 + i * 30), line, font=font(18), fill=MUTED)


def arrow(d, start, end):
    x, y = end
    d.line([start, end], fill=TEAL, width=4)
    d.polygon([(x, y), (x - 11, y - 7), (x - 11, y + 7)], fill=TEAL)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    stages = [
        ('01  Action', ['Bolus and meal levels', 'Acceptance rules apply']),
        ('02  Simulator', ['Physiology + delivery', 'Sensor + behavioral noise']),
        ('03  Step result', ['Observation, reward, cost', 'Termination flags + info']),
    ]
    for active in range(3):
        im, d = canvas('GlucoSim · agent–environment interaction',
                       'Conceptual animation — not a patient trajectory or benchmark.', 430)
        for i, (title, lines) in enumerate(stages):
            x = 42 + i * 383
            card(d, (x, 130, x + 350, 290), title, lines, active=i == active)
            if i < 2:
                arrow(d, (x + 355, 210), (x + 375, 210))
        d.line([(980, 305), (980, 335), (215, 335), (215, 308)], fill=TEAL, width=3)
        d.polygon([(215, 301), (208, 313), (222, 313)], fill=TEAL)
        d.text((395, 352), 'Choose the next action until the episode ends', font=font(18), fill=MUTED)
        frames.append(im)
    frames[0].save(OUT / 'simulation-loop.gif', save_all=True, append_images=frames[1:],
                   duration=1100, loop=0, optimize=False)


if __name__ == '__main__':
    main()
