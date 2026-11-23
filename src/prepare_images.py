import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from hopfield import viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--height", type=int, default=300)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out-dir", default="data_large")
    a = ap.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in exts)[: a.limit]
    rows = []
    for f in files:
        g = np.asarray(ImageOps.fit(Image.open(f).convert("L"), (a.width, a.height)), float)
        rows.append(np.where(g > np.median(g), 1, -1).ravel())  # próg = mediana -> zrównoważone wzorce
    X = np.array(rows, int)

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    name = f"cats_{a.height}x{a.width}"
    np.savetxt(out / f"{name}.csv", X, fmt="%d", delimiter=",")
    viz.show_grid(X[:16], (a.height, a.width), out / f"{name}_preview.png", ncols=8)
    print(f"{len(X)} bitmap -> {out / (name + '.csv')}")


if __name__ == "__main__":
    main()
