import argparse

from hopfield import experiments as ex

COMMANDS = {
    "stability": ex.exp_stability,
    "recovery": ex.exp_recovery,
    "compare": ex.exp_compare,
    "random": ex.exp_random_input,
    "stable-set": ex.exp_stable_set,
    "oscillation": ex.exp_oscillation,
    "large": ex.exp_large,
    "demo": ex.demo,
    "check-oja": ex.check_oja,
}


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("files", nargs="*", help="pliki CSV (domyślnie data/*.csv)")
    common.add_argument("--out", default="output")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--trials", type=int, default=30)
    common.add_argument("--reps", type=int, default=5)
    common.add_argument("--p-points", type=int, default=10)
    common.add_argument("--noise", type=float, default=0.1)
    common.add_argument("--modes", nargs="+", default=["sync", "async"])
    common.add_argument("--max-steps", type=int, default=100)
    common.add_argument("--oja-solver", choices=["iterative", "svd"], default="svd")
    common.add_argument("--oja-zero-diag", action="store_true")

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in [*COMMANDS, "all"]:
        sub.add_parser(name, parents=[common])
    s = sub.choices
    s["compare"].add_argument("--equalize", action="store_true", help="przytnij zbiory do równej liczności")
    s["large"].add_argument("--p-list", type=int, nargs="+", default=[2, 5, 10, 20])
    s["stable-set"].add_argument("--patience", type=int, default=2000)
    s["stable-set"].add_argument("--max-size", type=int, default=60)
    s["stable-set"].add_argument("--restarts", type=int, default=5)
    s["oscillation"].add_argument("--search-trials", type=int, default=50000)
    s["demo"].add_argument("--rule", choices=["hebb", "oja"], default="hebb")
    s["demo"].add_argument("--mode", choices=["sync", "async"], default="sync")
    s["demo"].add_argument("--kind", choices=["noise", "block"], default="noise")
    s["demo"].add_argument("--level", type=float, default=0.2)
    s["demo"].add_argument("--index", type=int, default=0)
    s["demo"].add_argument("--record-every", type=int, default=None)
    s["demo"].add_argument("--show", action="store_true")

    args = ap.parse_args()
    if args.cmd == "all":
        for c in ("stability", "recovery", "random", "stable-set", "oscillation"):
            args.cmd = c
            if c == "stable-set":
                args.patience, args.max_size, args.restarts = 2000, 60, 5
            if c == "oscillation":
                args.search_trials = 50000
            print(f"=== {c} ===")
            COMMANDS[c](args)
    else:
        COMMANDS[args.cmd](args)


if __name__ == "__main__":
    main()
