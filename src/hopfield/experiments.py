from collections import Counter
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data, viz
from .network import HopfieldNetwork

KINDS = {"noise": np.linspace(0, 0.5, 11), "block": np.linspace(0, 0.6, 7)}


# ---------- pomocnicze ----------
def variants(args):
    return {
        "hebb": ("hebb", {}),
        "oja": ("oja", {"zero_diag": args.oja_zero_diag, "solver": args.oja_solver}),
    }


def trained(X, shape, variant, seed=0):
    rule, kw = variant
    return HopfieldNetwork(shape, seed).train(X, rule, **kw)


def resolve_files(args):
    return args.files or sorted(glob("data/*.csv"))


def _load(path):
    X, shape = data.load_patterns(path)
    return Path(path).stem, X, shape


def _out(args, name):
    d = Path(args.out) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p_points(P, k):
    return np.unique(np.linspace(1, P, min(P, k)).astype(int))


def recovery(net, X, shape, kind, level, mode, trials, rng, max_steps=100):
    ok, ov = 0, []
    for _ in range(trials):
        j = rng.integers(len(X))
        r = net.recall(data.corrupt(X[j], shape, kind, level, rng), mode, max_steps)
        ok += np.array_equal(r.final, X[j])
        ov.append(r.final @ X[j] / net.n)
    return ok / trials, float(np.mean(ov))


def recovery_table(name, X, shape, args, rng):
    rows = []
    for label, v in variants(args).items():
        net = trained(X, shape, v)
        for mode in args.modes:
            for kind, levels in KINDS.items():
                for lv in levels:
                    s, o = recovery(net, X, shape, kind, lv, mode, args.trials, rng, args.max_steps)
                    rows.append(dict(dataset=name, rule=label, mode=mode, kind=kind, level=lv,
                                     success=s, overlap=o, p=len(X)))
    return pd.DataFrame(rows)


def plot_recovery(df, path, title=""):
    names = list(df.dataset.unique())
    fig, axs = plt.subplots(1, len(KINDS), figsize=(6 * len(KINDS), 4.5), squeeze=False)
    for ax, kind in zip(axs[0], KINDS):
        for (name, rule, mode), g in df[df.kind == kind].groupby(["dataset", "rule", "mode"]):
            ax.plot(g.level, g.success, color=f"C{names.index(name)}",
                    ls="-" if rule == "hebb" else "--", marker="o" if mode == "sync" else "s",
                    label=f"{name} | {rule} | {mode}")
        ax.set_xlabel("poziom uszkodzenia" + (" (ułamek bitów)" if kind == "noise" else " (ułamek pola)"))
        ax.set_ylabel("odzyskane oryginały")
        ax.set_title(f"{title} uszkodzenie: {kind}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axs[0][0].legend(fontsize=7)
    viz.save(fig, path)


# ---------- 1-3: stabilność wzorców, korelacje ----------
def exp_stability(args):
    out = _out(args, "stability")
    rng = np.random.default_rng(args.seed)
    files = resolve_files(args)
    rows = []
    fig, axs = plt.subplots(1, len(files), figsize=(5 * len(files), 4), squeeze=False)
    for ax, path in zip(axs[0], files):
        name, X, shape = _load(path)
        st = data.pattern_stats(X)
        for label, v in variants(args).items():
            m = trained(X, shape, v).stable_mask(X)
            rows.append(dict(dataset=name, rule=label, **st, stable=int(m.sum()), stable_frac=m.mean()))
        ps = _p_points(len(X), args.p_points)
        acc = {l: {p: [] for p in ps} for l in variants(args)}
        for p in ps:
            for _ in range(args.reps):
                S = data.subset(X, p, rng)
                for label, v in variants(args).items():
                    acc[label][p].append(trained(S, shape, v).stable_mask(S).mean())
        for label in acc:
            ax.plot(ps, [np.mean(acc[label][p]) for p in ps], marker="o", label=label)
        ax.axvline(0.138 * X.shape[1], color="gray", ls=":", label="0.138 n")
        ax.set(title=f"{name} (n={X.shape[1]})", xlabel="liczba wzorców p", ylabel="ułamek stabilnych")
        ax.legend()
        ax.grid(alpha=0.3)
        C = X @ X.T / X.shape[1]
        f2, a2 = plt.subplots(figsize=(5, 4.5))
        a2.set_title(f"{name}: korelacje wzorców")
        f2.colorbar(a2.imshow(C, cmap="coolwarm", vmin=-1, vmax=1), ax=a2)
        viz.save(f2, out / f"correlation_{name}.png")
        viz.show_grid(X[: min(len(X), 24)], shape, out / f"patterns_{name}.png")
    viz.save(fig, out / "stable_vs_p.png")
    pd.DataFrame(rows).to_csv(out / "stability.csv", index=False)
    print(pd.DataFrame(rows).to_string())


# ---------- 4: odzyskiwanie z uszkodzonego wejścia ----------
def exp_recovery(args):
    out = _out(args, "recovery")
    rng = np.random.default_rng(args.seed)
    files = resolve_files(args)
    loaded = [_load(f) for f in files]
    tables = []
    for name, X, shape in loaded:
        df = recovery_table(name, X, shape, args, rng)
        df.to_csv(out / f"recovery_{name}.csv", index=False)
        plot_recovery(df, out / f"recovery_{name}.png", name)
        tables.append(df)
    pd.concat(tables).to_csv(out / "recovery_all.csv", index=False)

    fig, axs = plt.subplots(1, len(loaded), figsize=(5 * len(loaded), 4), squeeze=False)
    rows = []
    for ax, (name, X, shape) in zip(axs[0], loaded):
        ps = _p_points(len(X), args.p_points)
        acc = {l: {p: [] for p in ps} for l in variants(args)}
        for p in ps:
            for _ in range(args.reps):
                S = data.subset(X, p, rng)
                for label, v in variants(args).items():
                    s, _ = recovery(trained(S, shape, v), S, shape, "noise", args.noise, "sync",
                                    args.trials, rng, args.max_steps)
                    acc[label][p].append(s)
        for label in acc:
            ys = [np.mean(acc[label][p]) for p in ps]
            ax.plot(ps, ys, marker="o", label=label)
            rows += [dict(dataset=name, rule=label, p=p, success=y) for p, y in zip(ps, ys)]
        ax.set(title=f"{name}, szum {args.noise}", xlabel="liczba wzorców p", ylabel="odzyskane oryginały",
               ylim=(-0.02, 1.02))
        ax.legend()
        ax.grid(alpha=0.3)
    viz.save(fig, out / "recovery_vs_p.png")
    pd.DataFrame(rows).to_csv(out / "recovery_vs_p.csv", index=False)


# ---------- 3, 5: porównanie dwóch zbiorów ----------
def exp_compare(args):
    files = resolve_files(args)
    assert len(files) == 2, "podaj dokładnie dwa pliki"
    out = _out(args, "compare")
    rng = np.random.default_rng(args.seed)
    sets = [_load(f) for f in files]
    p = min(len(s[1]) for s in sets) if args.equalize else None
    tables, stats = [], []
    for name, X, shape in sets:
        if p:
            X = data.subset(X, p, rng)
        tables.append(recovery_table(name, X, shape, args, rng))
        stats.append(dict(dataset=name, **data.pattern_stats(X)))
        for label, v in variants(args).items():
            stats[-1][f"stable_{label}"] = trained(X, shape, v).stable_mask(X).mean()
    df = pd.concat(tables)
    tag = "_".join(s[0] for s in sets)
    df.to_csv(out / f"compare_{tag}.csv", index=False)
    pd.DataFrame(stats).to_csv(out / f"compare_stats_{tag}.csv", index=False)
    plot_recovery(df, out / f"compare_{tag}.png", "porównanie")
    print(pd.DataFrame(stats).to_string())


# ---------- 6: losowe wejście ----------
def exp_random_input(args):
    out = _out(args, "random_input")
    rng = np.random.default_rng(args.seed)
    rows = []
    for path in resolve_files(args):
        name, X, shape = _load(path)
        n = X.shape[1]
        for label, v in variants(args).items():
            net = trained(X, shape, v)
            for mode in args.modes:
                cnt, finals = Counter(), []
                for _ in range(args.trials):
                    r = net.recall(rng.choice([-1.0, 1.0], n), mode, args.max_steps)
                    ov = X @ r.final
                    if r.status == "cycle":
                        c = "cykl"
                    elif r.status == "max_steps":
                        c = "brak zbieżności"
                    elif (ov == n).any():
                        c = "wzorzec"
                    elif (ov == -n).any():
                        c = "negatyw wzorca"
                    else:
                        c = "stan pozorny"
                    cnt[c] += 1
                    finals.append(r.final)
                rows.append(dict(dataset=name, rule=label, mode=mode,
                                 **{k: cnt[k] / args.trials for k in
                                    ["wzorzec", "negatyw wzorca", "stan pozorny", "cykl", "brak zbieżności"]}))
                viz.show_grid(finals[:8], shape, out / f"finals_{name}_{label}_{mode}.png", ncols=8)
    df = pd.DataFrame(rows)
    df.to_csv(out / "random_input.csv", index=False)
    d = df.assign(key=df.dataset + "|" + df.rule + "|" + df["mode"]).set_index("key").drop(
        columns=["dataset", "rule", "mode"])
    ax = d.plot(kind="bar", stacked=True, figsize=(max(6, 0.6 * len(d)), 5))
    ax.set_ylabel("ułamek losowych wejść")
    viz.save(ax.figure, out / "random_input.png")
    print(df.to_string())


# ---------- 7: największy zbiór 5x5 ze wszystkimi wzorcami stabilnymi ----------
def greedy_stable_set(variant, shape, rng, patience, max_size):
    n = shape[0] * shape[1]
    S, fails = np.empty((0, n)), 0
    while len(S) < max_size and fails < patience:
        c = rng.choice([-1.0, 1.0], n)
        if len(S) and np.abs(S @ c).max() == n:
            fails += 1
            continue
        T = np.vstack([S, c])
        if trained(T, shape, variant).stable_mask(T).all():
            S, fails = T, 0
        else:
            fails += 1
    return S


def exp_stable_set(args):
    out = _out(args, "stable_set")
    rng = np.random.default_rng(args.seed)
    shape = (5, 5)
    vs = dict(variants(args))
    vs["oja (bez diagonali)"] = ("oja", {"zero_diag": True, "solver": "svd"})
    rows = []
    for label, v in vs.items():
        best = max((greedy_stable_set(v, shape, rng, args.patience, args.max_size)
                    for _ in range(args.restarts)), key=len)
        rows.append(dict(rule=label, size=len(best), capped=len(best) >= args.max_size))
        np.savetxt(out / f"set_{label.split()[0]}_{len(best)}.csv", best, fmt="%d", delimiter=",")
        viz.show_grid(best[:40], shape, out / f"set_{label.replace(' ', '_')}.png", ncols=10)
    pd.DataFrame(rows).to_csv(out / "stable_set.csv", index=False)
    print(pd.DataFrame(rows).to_string())


# ---------- 8: oscylacja ----------
def exp_oscillation(args):
    out = _out(args, "oscillation")
    rng = np.random.default_rng(args.seed)
    net = HopfieldNetwork((1, 2)).train(np.array([[1.0, -1.0]]), "hebb")
    x0 = np.array([1.0, 1.0])
    for mode in ("sync", "async"):
        r = net.recall(x0, mode, 10)
        print(f"2 neurony, {mode}: {r.status}, okres {r.period}, ścieżka {[h.tolist() for h in r.history]}")
        viz.plot_history(r.history, (1, 2), out / f"two_neurons_{mode}.png", ncols=6)

    shape = (5, 5)
    for _ in range(args.search_trials):
        S = rng.choice([-1.0, 1.0], (int(rng.integers(2, 6)), 25))
        net = HopfieldNetwork(shape).train(S, "hebb")
        x0 = rng.choice([-1.0, 1.0], 25)
        r = net.recall(x0, "sync", 50)
        if r.status == "cycle" and r.period == 2:
            np.savetxt(out / "training_set_5x5.csv", S, fmt="%d", delimiter=",")
            np.savetxt(out / "input_5x5.csv", x0[None], fmt="%d", delimiter=",")
            viz.show_grid(S, shape, out / "training_set_5x5.png", ncols=5)
            viz.plot_history(r.history, shape, out / "cycle_5x5_sync.png", ncols=5)
            viz.plot_history(net.recall(x0, "async", 50).history, shape, out / "async_5x5.png")
            print(f"Znaleziono oscylację 5x5 dla p={len(S)}")
            return
    print("Nie znaleziono oscylacji 5x5; zwiększ --search-trials lub zmień --seed")


# ---------- 9: duże bitmapy ----------
def exp_large(args):
    out = _out(args, "large")
    rng = np.random.default_rng(args.seed)
    rows = []
    for path in resolve_files(args):
        name, X, shape = _load(path)
        print(name, X.shape, flush=True)
        st = data.pattern_stats(X)
        for p in [q for q in args.p_list if q <= len(X)]:
            S = data.subset(X, p, rng)
            for label, v in variants(args).items():
                net = trained(S, shape, v)
                row = dict(dataset=name, rule=label, p=p, stable_frac=net.stable_mask(S).mean(),
                           mean_abs_overlap=st["mean_abs_overlap"])
                for kind, lv in (("noise", 0.2), ("noise", 0.4), ("block", 0.3)):
                    for mode in args.modes:
                        s, o = recovery(net, S, shape, kind, lv, mode, args.trials, rng, args.max_steps)
                        row[f"{kind}{lv}_{mode}"] = s
                rows.append(row)
                print(row, flush=True)
        p = min(max(args.p_list), len(X))
        S = data.subset(X, p, rng)
        for label, v in variants(args).items():
            net = trained(S, shape, v)
            for kind, lv in (("noise", 0.3), ("block", 0.3)):
                for mode in args.modes:
                    r = net.recall(data.corrupt(S[0], shape, kind, lv, rng), mode, args.max_steps)
                    viz.plot_history(r.history, shape, out / f"example_{name}_{label}_{kind}_{mode}.png",
                                     max_frames=8, ncols=4)
    pd.DataFrame(rows).to_csv(out / "large.csv", index=False)


# ---------- demo i kontrola reguły Oji ----------
def demo(args):
    name, X, shape = _load(resolve_files(args)[0])
    v = variants(args)[args.rule]
    net = trained(X, shape, v)
    rng = np.random.default_rng(args.seed)
    x0 = data.corrupt(X[args.index], shape, args.kind, args.level, rng)
    r = net.recall(x0, args.mode, args.max_steps, args.record_every)
    print(r.status, "kroki:", r.steps, "okres:", r.period, "odzyskano:", np.array_equal(r.final, X[args.index]))
    out = _out(args, "demo")
    tag = f"{name}_{args.rule}_{args.mode}"
    viz.plot_history(r.history, shape, out / f"{tag}.png")
    viz.plot_energy(net, r.history, out / f"{tag}_energy.png")
    viz.animate(r.history, shape, None if args.show else out / f"{tag}.gif", net=net)


def check_oja(args):
    name, X, shape = _load(resolve_files(args)[0])
    nets = {s: HopfieldNetwork(shape, args.seed).train(X, "oja", solver=s, zero_diag=args.oja_zero_diag)
            for s in ("iterative", "svd")}
    P = {s: n.A @ n.A.T for s, n in nets.items()}
    print("||P_iter - P_svd|| / ||P_svd|| =", np.linalg.norm(P["iterative"] - P["svd"]) / np.linalg.norm(P["svd"]))
    for s, n in nets.items():
        print(s, "stabilne:", n.stable_mask(X).mean())
