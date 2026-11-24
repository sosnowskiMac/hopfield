
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Tabela i wykres dla eksperymentu dużych bitmap"
)
parser.add_argument(
    "--csv",
    default="output_moje/large/large.csv"
)
parser.add_argument(
    "--out",
    default="output_moje/large/moj_wykres.png"
)
parser.add_argument(
    "--trials",
    type=int,
    default=4
)

args = parser.parse_args()

# Wczytanie wyników
src = Path(args.csv)

if not src.exists():
    parser.error(
        f"Nie znaleziono pliku {src}. "
        "Najpierw uruchom eksperyment large."
    )

df = pd.read_csv(src)

cols = [
    "rule",
    "p",
    "stable_frac",
    "noise0.2_sync",
    "noise0.4_sync",
    "block0.3_sync"
]

missing = [c for c in cols if c not in df.columns]

if missing:
    parser.error(
        "Brakuje kolumn: " + ", ".join(missing)
    )

# Tabela z wynikami
print("\nWYNIKI EKSPERYMENTU\n")

print(
    f'{"Reguła":<8} {"p":>3} {"Stabilność":>12} '
    f'{"Szum 20%":>11} {"Szum 40%":>11} {"Blok 30%":>11}'
)

for _, r in df.sort_values(["p", "rule"]).iterrows():

    p = int(r["p"])

    stable = f"{round(r['stable_frac'] * p)}/{p}"

    rates = [
        f"{round(r[c] * args.trials)}/{args.trials}"
        for c in cols[3:]
    ]

    print(
        f"{r['rule']:<8} {p:>3} {stable:>12} "
        f"{rates[0]:>11} {rates[1]:>11} {rates[2]:>11}"
    )

# Generowanie wykresów
fig, (left, right) = plt.subplots(
    1, 2, figsize=(13, 4.8), constrained_layout=True
)

styles = {
    "hebb": "o-",
    "oja": "s--"
}

for rule, group in df.groupby("rule"):

    group = group.sort_values("p")

    # Stabilność
    left.plot(
        group["p"],
        group["stable_frac"],
        styles.get(rule, "o-"),
        label=rule
    )

    # Rekonstrukcja
    for key, title, ls in [
        ("noise0.2_sync", "Szum 20%", "-"),
        ("noise0.4_sync", "Szum 40%", "--"),
        ("block0.3_sync", "Blok 30%", ":")
    ]:

        right.plot(
            group["p"],
            group[key],
            marker="o" if rule == "hebb" else "s",
            linestyle=ls,
            label=f"{rule}: {title}"
        )

left.set(
    title="Stabilność wzorców",
    xlabel="Liczba wzorców p",
    ylabel="Ułamek stabilnych"
)

right.set(
    title="Rekonstrukcja uszkodzonych obrazów",
    xlabel="Liczba wzorców p",
    ylabel="Odsetek poprawnych rekonstrukcji"
)

for ax in (left, right):
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sorted(df["p"].unique()))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

# Zapis wykresu
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(out, dpi=170)

print(f"\nWykres zapisano: {out}")

plt.show()
