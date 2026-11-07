"""Turn results/*.npz into tables and figures.
  python -m src.analysis --results results --cache-dir CACHE --focus-n 64
Writes: summary.csv, summary.md, per_camera.csv and fig_*.png in --results."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from .metrics import accuracy, macro_f1

COL = {"bn": "#d95f02", "tent": "#1b9e77", "none": "#555555"}
LABEL = {"bn": "BN statistics update", "tent": "Entropy min. (TENT)", "none": "No adaptation"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def load_runs(results_dir, split):
    runs = []
    for f in sorted(glob.glob(str(Path(results_dir) / "*" / "*.npz"))):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        if meta["split"] != split:
            continue
        runs.append((meta, pd.DataFrame({k: d[k] for k in ("camera", "idx", "y", "pred", "base")})))
    return runs


def mean_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) < 2:
        return float(x.mean()), np.nan
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)) * stats.t.ppf(0.975, len(x) - 1))


def common_cameras(runs):
    """Cameras usable at every numeric n (so curves over n compare the same cameras)."""
    sets = {}
    for m, df in runs:
        if m["n"] != "all" and m["empty_frac"] is None and m["scope"] == "all":
            sets.setdefault((m["backbone"], m["n"]), set()).update(df.camera.unique().tolist())
    out = {}
    for (bb, _), s in sets.items():
        out[bb] = s if bb not in out else out[bb] & s
    return out


def run_table(runs, common):
    rows = []
    for m, df in runs:
        if m["n"] != "all" and m["empty_frac"] is None and m["scope"] == "all" and m["backbone"] in common:
            df = df[df.camera.isin(common[m["backbone"]])]
        rows.append({**{k: m[k] for k in ("backbone", "method", "scope", "n", "seed", "empty_frac")},
                     "bn_prior": float(m.get("bn_prior", 0.0)),
                     "acc": accuracy(df.y, df.pred), "f1": macro_f1(df.y, df.pred),
                     "base_acc": accuracy(df.y, df.base), "base_f1": macro_f1(df.y, df.base),
                     "n_cams": df.camera.nunique(), "n_eval": len(df)})
    t = pd.DataFrame(rows)
    t["d_acc"], t["d_f1"] = t.acc - t.base_acc, t.f1 - t.base_f1
    t["empty_frac"] = t.empty_frac.astype(object).where(t.empty_frac.notna(), "random")
    return t


def summarise(t):
    keys = ["backbone", "method", "bn_prior", "scope", "n", "empty_frac"]
    out = []
    for k, g in t.groupby(keys, dropna=False):
        r = dict(zip(keys, k))
        for c in ("acc", "f1", "base_acc", "base_f1", "d_acc", "d_f1"):
            r[c], r[c + "_ci"] = mean_ci(g[c])
        r["seeds"], r["cams"] = g.seed.nunique(), int(g.n_cams.max())
        out.append(r)
    s = pd.DataFrame(out)
    s["n_sort"] = pd.to_numeric(s.n, errors="coerce").fillna(1e9)
    return s.sort_values(["backbone", "scope", "method", "bn_prior", "n_sort"]).drop(columns="n_sort").reset_index(drop=True)


def md_table(df, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fig_curve(s, path):
    """Paired change in macro F1 (adapted minus no adaptation, same held-out images) vs. n."""
    main = s[(s.scope == "all") & (s.empty_frac == "random")]
    bbs = sorted(main.backbone.unique())
    fig, axes = plt.subplots(1, len(bbs), figsize=(4.6 * len(bbs), 3.6), squeeze=False)
    for ax, bb in zip(axes[0], bbs):
        g = main[main.backbone == bb]
        num = g[g.n != "all"].copy()
        num["nn"] = num.n.astype(int)
        ax.axhline(0, color=COL["none"], lw=1, label=LABEL["none"])
        for (meth, pr), gg in num.groupby(["method", "bn_prior"]):
            gg = gg.sort_values("nn")
            lab = LABEL[meth] + (f" (prior {pr:g})" if pr > 0 else "")
            ax.errorbar(gg.nn, gg.d_f1, yerr=gg.d_f1_ci, color=COL[meth], marker="o", ms=3, capsize=2, label=lab,
                        ls="-" if pr == 0 else "--", alpha=1 if pr == 0 else 0.7)
            orc = g[(g.n == "all") & (g.method == meth) & (g.bn_prior == pr)]
            if len(orc):
                ax.axhline(orc.d_f1.iloc[0], color=COL[meth], ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(num.nn.unique()))
        ax.set_xticklabels(sorted(num.nn.unique()))
        ax.set_xlabel("unlabeled images from the new camera (n)")
        ax.set_ylabel("change in macro F1 vs. no adaptation")
        ax.set_title(bb)
        ax.legend(frameon=False, fontsize=8)
    fig.text(0.99, 0.01, "dotted = oracle (all images of the camera); bars = 95% CI over seeds", ha="right", fontsize=7, color="#666")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def per_camera(runs, focus_n, focus_prior=0.0):
    rows = []
    for m, df in runs:
        if m["n"] != str(focus_n) or m["empty_frac"] is not None or m["scope"] != "all":
            continue
        if float(m.get("bn_prior", 0.0)) != focus_prior:
            continue
        for cam, g in df.groupby("camera"):
            rows.append({"backbone": m["backbone"], "method": m["method"], "seed": m["seed"], "camera": cam,
                         "acc": accuracy(g.y, g.pred), "base_acc": accuracy(g.y, g.base), "n_eval": len(g)})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    t = t.groupby(["backbone", "method", "camera"], as_index=False).agg(acc=("acc", "mean"), base_acc=("base_acc", "mean"),
                                                                       n_eval=("n_eval", "mean"))
    t["delta"] = t.acc - t.base_acc
    return t


def add_conditions(pc, cache_dir, split):
    from .common import Cache
    c = Cache(cache_dir, split)
    br, em = {}, {}
    for cam, ids in c.camera_groups().items():
        sub = c.get(ids[:60])
        br[cam] = float(np.asarray(sub, dtype=np.float32).mean() / 255)
        em[cam] = float((c.y[ids] == 0).mean())
    pc["brightness"] = pc.camera.map(br)
    pc["empty_frac"] = pc.camera.map(em)
    return pc


def fig_percamera(pc, path, focus_n):
    keys = sorted(pc.groupby(["backbone", "method"]).groups)
    fig, ax = plt.subplots(figsize=(1.9 * len(keys) + 1.5, 3.6))
    data = [pc[(pc.backbone == b) & (pc.method == m)].delta.values * 100 for b, m in keys]
    ax.axhline(0, color="#888", lw=0.8)
    ax.boxplot(data, showfliers=False)
    for i, d in enumerate(data, 1):
        ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(d)), d, s=8, color=COL[keys[i - 1][1]], alpha=0.7)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([f"{b}\n{m}" for b, m in keys], fontsize=7)
    ax.set_ylabel("per-camera accuracy change (points)")
    ax.set_title(f"Which cameras gain or lose (n={focus_n}); each dot = one camera")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_conditions(pc, path, focus_n):
    if "brightness" not in pc:
        return []
    keys = sorted(pc.groupby(["backbone", "method"]).groups)
    fig, axes = plt.subplots(2, len(keys), figsize=(2.9 * len(keys), 5.2), squeeze=False)
    corr = []
    for j, (b, m) in enumerate(keys):
        g = pc[(pc.backbone == b) & (pc.method == m)]
        for i, (col, lab) in enumerate((("brightness", "mean brightness"), ("empty_frac", "share of empty frames"))):
            ax = axes[i][j]
            ax.scatter(g[col], g.delta * 100, s=10, color=COL[m])
            ax.axhline(0, color="#888", lw=0.8)
            rho = stats.spearmanr(g[col], g.delta)[0] if len(g) > 3 else np.nan
            corr.append({"backbone": b, "method": m, "condition": col, "spearman": rho})
            ax.set_title(f"{b}/{m}  rho={rho:.2f}", fontsize=8)
            ax.set_xlabel(lab)
            if j == 0:
                ax.set_ylabel("accuracy change (points)")
    fig.suptitle(f"Does the gain depend on image conditions? (n={focus_n})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return corr


def fig_empty(s, path):
    g = s[(s.scope == "all") & (s.n == "64") & (s.empty_frac != "random") & (s.bn_prior == 0)]
    if g.empty:
        return
    g = g.assign(ef=g.empty_frac.astype(float))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for (bb, meth), gg in g.groupby(["backbone", "method"]):
        gg = gg.sort_values("ef")
        ax.errorbar(gg.ef, gg.d_f1, yerr=gg.d_f1_ci, marker="o", ms=3, capsize=2, label=f"{bb}/{meth}",
                    color=COL[meth], ls="-" if bb == "resnet50" else "--")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("share of empty frames in the adapt batch (n=64)")
    ax.set_ylabel("change in macro F1")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_seen_unseen(res_dir, path):
    f = Path(res_dir) / "baseline.json"
    if not f.exists():
        return
    b = json.load(open(f))
    bbs = list(b)
    x = np.arange(len(bbs))
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.bar(x - 0.18, [b[k]["id_test"]["macro_f1"] for k in bbs], 0.36, label="seen cameras", color="#7570b3")
    ax.bar(x + 0.18, [b[k]["test"]["macro_f1"] for k in bbs], 0.36, label="unseen cameras", color="#e7298a")
    ax.set_xticks(x)
    ax.set_xticklabels(bbs)
    ax.set_ylabel("macro F1 (no adaptation)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def camera_bootstrap(pc, B=2000):
    rng = np.random.default_rng(0)
    rows = []
    for (b, m), g in pc.groupby(["backbone", "method"]):
        d = g.delta.values
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
        rows.append({"backbone": b, "method": m, "cams": len(d), "mean_delta_acc": d.mean(),
                     "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                     "share_cams_improved": float((d > 0).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--focus-n", type=int, default=64)
    ap.add_argument("--focus-prior", type=float, default=0.0, help="BN prior used for the per-camera analysis")
    a = ap.parse_args()
    R = Path(a.results)
    runs = load_runs(R, a.split)
    if not runs:
        raise SystemExit("no result files found")
    common = common_cameras(runs)
    t = run_table(runs, common)
    s = summarise(t)
    t.to_csv(R / "runs.csv", index=False)
    s.to_csv(R / "summary.csv", index=False)

    cols = ["backbone", "method", "bn_prior", "n", "base_f1", "f1", "d_f1", "d_f1_ci", "base_acc", "acc", "d_acc", "d_acc_ci", "cams", "seeds"]
    md = ["# Results summary (unseen cameras)\n",
          "F1 = macro F1, acc = accuracy, d_* = change vs. no adaptation on the same held-out images, *_ci = 95% t-interval over seeds.",
          "`all` = oracle (transductive upper bound).\n",
          "## Main experiment (adapt all norm layers, random adapt batch)\n",
          md_table(s[(s.scope == "all") & (s.empty_frac == "random")], cols)]
    sc = s[(s.scope == "last") & (s.empty_frac == "random")]
    if len(sc):
        md += ["\n## Ablation: only the last stage/blocks are adapted\n", md_table(sc, cols)]
    em = s[s.empty_frac != "random"]
    if len(em):
        md += ["\n## Ablation: share of empty frames in the adapt batch\n", md_table(em, ["backbone", "method", "bn_prior", "n", "empty_frac"] + cols[4:])]

    pc = per_camera(runs, a.focus_n, a.focus_prior)
    if len(pc):
        if a.cache_dir:
            pc = add_conditions(pc, a.cache_dir, a.split)
        pc.to_csv(R / "per_camera.csv", index=False)
        fig_percamera(pc, R / "fig_per_camera.png", a.focus_n)
        corr = fig_conditions(pc, R / "fig_conditions.png", a.focus_n)
        bt = camera_bootstrap(pc)
        md += [f"\n## Per camera (n={a.focus_n})\n", md_table(bt, list(bt.columns))]
        if corr:
            md += ["\n### Spearman correlation: accuracy change vs. image condition\n", md_table(pd.DataFrame(corr), ["backbone", "method", "condition", "spearman"])]
    fig_curve(s, R / "fig_curve.png")
    fig_empty(s, R / "fig_empty_frac.png")
    fig_seen_unseen(R, R / "fig_seen_unseen.png")
    (R / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
