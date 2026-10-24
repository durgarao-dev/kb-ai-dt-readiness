"""KAID: knowledge-guided investment decision for digital-transformation readiness.

Reproducible short-communication experiment. One command rewrites the figures
and results.json that PAPER.md reports:

    python kb_dt_model.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import binomtest, wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

SEED = 20261024
N_ENTERPRISES = 360
N_FOLDS = 5
N_BOOT = 4000
DELTA = 0.15
SENSITIVITY = (0.10, 0.15, 0.20)
INDICATOR_NOISE = 0.14
AUDIT_NOISE = 0.10
PAIR_MAIN = 4.0
PAIR_SECOND = 0.9

ASSETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Know-how", ("Training intensity", "Process innovation", "R&D intensity")),
    ("Know-who", ("Customer cooperation", "Supplier cooperation", "University cooperation")),
    ("Know-what", ("Product innovation", "Innovation sales share")),
    ("Know-why", ("Scientific sourcing", "Research-project share")),
)
ASSET_NAMES = [name for name, _ in ASSETS]
DISTRACTORS = (
    "Plant age",
    "Fleet size",
    "Unrelated certifications",
    "Office count",
    "Vehicle age",
    "Storage area",
)
METHOD_ORDER = (
    "Lowest score",
    "Raw gradient boosting",
    "Equal-weight knowledge rules",
    "KAID",
)


@dataclass(frozen=True)
class Population:
    indicators: np.ndarray
    distractors: np.ndarray
    scores: np.ndarray
    latents: np.ndarray
    audit: np.ndarray
    asset_slices: tuple[slice, ...]


def readiness_index(latents: np.ndarray) -> np.ndarray:
    """Two complementary pairs. The first pair is weighted more heavily."""
    how, who, what, why = (latents[:, i] for i in range(4))
    return PAIR_MAIN * how * who + PAIR_SECOND * what * why


def simulate(rng: np.random.Generator) -> Population:
    latents = rng.beta(2.0, 2.0, size=(N_ENTERPRISES, 4))
    columns: list[np.ndarray] = []
    slices: list[slice] = []
    cursor = 0
    for asset_i, (_, items) in enumerate(ASSETS):
        start = cursor
        for _ in items:
            noisy = latents[:, asset_i] + rng.normal(0.0, INDICATOR_NOISE, N_ENTERPRISES)
            columns.append(np.clip(noisy, 0.0, 1.0))
            cursor += 1
        slices.append(slice(start, cursor))
    indicators = np.column_stack(columns)
    scores = np.column_stack([indicators[:, sl].mean(axis=1) for sl in slices])
    distractors = rng.beta(2.0, 2.0, size=(N_ENTERPRISES, len(DISTRACTORS)))
    audit = readiness_index(latents) + rng.normal(0.0, AUDIT_NOISE, N_ENTERPRISES)
    return Population(indicators, distractors, scores, latents, audit, tuple(slices))


def make_boosting() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.06,
        max_iter=300,
        min_samples_leaf=12,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=SEED,
    )


def pair_features(scores: np.ndarray) -> np.ndarray:
    return np.column_stack([scores[:, 0] * scores[:, 1], scores[:, 2] * scores[:, 3]])


def oof_kaid(scores: np.ndarray, audit: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Learn only the two knowledge-pair weights, then fund the best one-asset move."""
    features = pair_features(scores)
    choice = np.zeros(len(audit), dtype=int)
    predicted = np.zeros(len(audit), dtype=float)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in folds.split(features):
        model = LinearRegression()
        model.fit(features[train_idx], audit[train_idx])
        held = scores[test_idx]
        baseline = model.predict(pair_features(held))
        predicted[test_idx] = baseline
        lifts = np.zeros((len(test_idx), 4))
        for asset_i in range(4):
            funded = held.copy()
            funded[:, asset_i] = np.minimum(1.0, funded[:, asset_i] + delta)
            lifts[:, asset_i] = model.predict(pair_features(funded)) - baseline
        choice[test_idx] = lifts.argmax(axis=1)
    return choice, predicted


def oof_action(
    features: np.ndarray,
    audit: np.ndarray,
    groups: tuple[np.ndarray, ...],
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit out of fold and choose the asset whose funded increase most raises the audit prediction."""
    choice = np.zeros(len(audit), dtype=int)
    predicted = np.zeros(len(audit), dtype=float)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in folds.split(features):
        model = make_boosting()
        model.fit(features[train_idx], audit[train_idx])
        held = features[test_idx]
        baseline = model.predict(held)
        predicted[test_idx] = baseline
        lifts = np.zeros((len(test_idx), len(groups)))
        for asset_i, columns in enumerate(groups):
            funded = held.copy()
            funded[:, columns] = np.minimum(1.0, funded[:, columns] + delta)
            lifts[:, asset_i] = model.predict(funded) - baseline
        choice[test_idx] = lifts.argmax(axis=1)
    return choice, predicted


def rule_action(scores: np.ndarray, delta: float) -> np.ndarray:
    """Equal-weight pair rules. Coefficients of the two products are not used."""
    base = scores[:, 0] * scores[:, 1] + scores[:, 2] * scores[:, 3]
    lifts = np.zeros_like(scores)
    for asset_i in range(4):
        funded = scores.copy()
        funded[:, asset_i] = np.minimum(1.0, funded[:, asset_i] + delta)
        lifts[:, asset_i] = (
            funded[:, 0] * funded[:, 1] + funded[:, 2] * funded[:, 3] - base
        )
    return lifts.argmax(axis=1)


def true_lifts(latents: np.ndarray, delta: float) -> np.ndarray:
    base = readiness_index(latents)
    lifts = np.zeros_like(latents)
    for asset_i in range(4):
        funded = latents.copy()
        funded[:, asset_i] = np.minimum(1.0, funded[:, asset_i] + delta)
        lifts[:, asset_i] = readiness_index(funded) - base
    return lifts


def realized_lift(latents: np.ndarray, choice: np.ndarray, delta: float) -> np.ndarray:
    funded = latents.copy()
    funded[np.arange(len(choice)), choice] = np.minimum(
        1.0, funded[np.arange(len(choice)), choice] + delta
    )
    return readiness_index(funded) - readiness_index(latents)


def mean_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, list[float]]:
    draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_diff_interval(proposed: np.ndarray, baseline: np.ndarray, rng: np.random.Generator) -> list[float]:
    difference = proposed - baseline
    draws = rng.integers(0, len(difference), size=(N_BOOT, len(difference)))
    means = difference[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def wilcoxon_p(proposed: np.ndarray, baseline: np.ndarray) -> float:
    difference = proposed - baseline
    if np.allclose(difference, 0.0):
        return 1.0
    return float(wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue)


def mcnemar_p(proposed_hit: np.ndarray, baseline_hit: np.ndarray) -> float:
    proposed_only = int(np.sum(proposed_hit & ~baseline_hit))
    baseline_only = int(np.sum(~proposed_hit & baseline_hit))
    discordant = proposed_only + baseline_only
    if discordant == 0:
        return 1.0
    return float(binomtest(proposed_only, discordant, 0.5, alternative="two-sided").pvalue)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        corrected = min(1.0, (count - rank) * p_values[index])
        running = max(running, corrected)
        adjusted[index] = running
    return [float(value) for value in adjusted]


def select_case(
    lifts: dict[str, np.ndarray],
    choices: dict[str, np.ndarray],
    oracle: np.ndarray,
) -> int:
    """Median KAID gain among enterprises named by KAID and missed by both baselines."""
    advantage = lifts["KAID"] - lifts["Raw gradient boosting"]
    mask = (
        (choices["KAID"] == oracle)
        & (choices["Raw gradient boosting"] != oracle)
        & (choices["Equal-weight knowledge rules"] != oracle)
    )
    if not np.any(mask):
        mask = (choices["KAID"] == oracle) & (choices["Equal-weight knowledge rules"] != oracle)
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        return int(np.argsort(lifts["KAID"])[len(lifts["KAID"]) // 2])
    median_advantage = np.median(advantage[candidates])
    return int(candidates[np.argmin(np.abs(advantage[candidates] - median_advantage))])


def _panel(ax, x, y, w, h, face, title, body) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.04",
            facecolor=face,
            edgecolor="#1C2B39",
            linewidth=0.8,
        )
    )
    ax.text(x + 0.12, y + h - 0.22, title, ha="left", va="top", fontsize=8.5, fontweight="bold", color="#1C2B39", fontfamily="serif")
    ax.text(x + 0.12, y + h - 0.52, body, ha="left", va="top", fontsize=7.4, color="#243444", fontfamily="serif", linespacing=1.38)


def plot_framework(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "mathtext.fontset": "dejavuserif"})
    fig, ax = plt.subplots(figsize=(7.60, 3.25))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10.05)
    ax.set_ylim(0, 3.55)
    ax.axis("off")
    _panel(
        ax, 0.12, 0.48, 2.95, 2.85, "#F4F7FB",
        "(a)  Knowledge layer",
        "Ten indicators become\n"
        "four observed scores.\n\n"
        r"$P_1$ = know-how $\times$ know-who" + "\n"
        r"$P_2$ = know-what $\times$ know-why" + "\n\n"
        "Six distractors are\n"
        "excluded from both products.",
    )
    _panel(
        ax, 3.55, 0.48, 2.95, 2.85, "#FBF6EE",
        "(b)  Audit model",
        "Out-of-fold least squares\n"
        r"$\hat{A} = \hat{c} + \hat{a}P_1 + \hat{b}P_2$" + "\n\n"
        r"$\hat{a}$ and $\hat{b}$ are the" + "\n"
        "learned pair weights.\n\n"
        "Raw indicators do not\n"
        "enter the regression.",
    )
    _panel(
        ax, 6.98, 0.48, 2.95, 2.85, "#F3F8F4",
        "(c)  One-asset investment",
        "Raise one score by 0.15\n"
        "and cap it at 1.\n\n"
        "Fund the asset with the\n"
        "largest predicted gain.\n\n"
        "The output is the name\n"
        "of that single asset.",
    )
    for start, end in ((3.09, 3.52), (6.52, 6.95)):
        ax.add_patch(
            FancyArrowPatch(
                (start, 1.90),
                (end, 1.90),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=0.9,
                color="#1C2B39",
            )
        )
    ax.text(
        5.02, 0.18,
        "The funded asset is scored by the increase in the true readiness index R.",
        ha="center", va="center", fontsize=7.2, color="#5C6B7A", fontfamily="serif",
    )
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.04, top=0.98)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_results(
    path: str,
    summary: list[dict],
    ceiling: float,
    case: dict,
) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(
        1, 2, figsize=(7.48, 3.55), gridspec_kw={"width_ratios": [1.02, 1.18]}
    )
    fig.patch.set_facecolor("white")
    colors = {
        "Lowest score": "#7A7A7A",
        "Raw gradient boosting": "#E69F00",
        "Equal-weight knowledge rules": "#009E73",
        "KAID": "#0072B2",
    }
    short = {
        "Lowest score": "Lowest score",
        "Raw gradient boosting": "Raw boosting",
        "Equal-weight knowledge rules": "Equal-weight rules",
        "KAID": "KAID",
    }

    ax = axes[0]
    labels = [row["method"] for row in summary]
    means = [row["mean_lift"] for row in summary]
    lows = [row["mean_lift"] - row["lift_ci"][0] for row in summary]
    highs = [row["lift_ci"][1] - row["mean_lift"] for row in summary]
    y_pos = np.arange(len(labels))[::-1]
    ax.barh(
        y_pos,
        means,
        color=[colors[name] for name in labels],
        height=0.62,
        xerr=[lows, highs],
        capsize=2.5,
        ecolor="#1C2B39",
        error_kw={"linewidth": 0.7},
    )
    ax.axvline(ceiling, color="#4C4C4C", ls="--", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean true readiness lift")
    ax.set_title("(a)", loc="left", fontsize=9, color="#1C2B39", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 0.49)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_ylim(-0.55, 4.35)
    ax.text(
        ceiling - 0.012,
        3.72,
        f"Best attainable\n{ceiling:.3f}",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="#333333",
        linespacing=1.15,
    )

    ax = axes[1]
    observed = np.asarray(case["observed"], dtype=float)
    latent = np.asarray(case["latent"], dtype=float)
    y_pos = np.arange(4)
    ax.hlines(y_pos, 0, observed, color="#D5DCE3", lw=4, zorder=1)
    ax.scatter(observed, y_pos, s=28, color="#0072B2", zorder=3, label="Observed score")
    ax.scatter(
        latent, y_pos, s=26, facecolors="none", edgecolors="#1C2B39",
        linewidths=0.9, zorder=3, label="True latent",
    )
    ax.axvline(1.0, color="#E2E6EA", lw=0.6, zorder=0)
    markers = {
        "KAID": "D",
        "Equal-weight knowledge rules": "s",
        "Raw gradient boosting": "^",
        "Lowest score": "o",
    }
    margin_x = {
        "Lowest score": 1.10,
        "Raw gradient boosting": 1.20,
        "Equal-weight knowledge rules": 1.30,
        "KAID": 1.40,
    }
    for method in METHOD_ORDER:
        asset_i = case["choices"][method]
        ax.scatter(
            margin_x[method],
            asset_i,
            marker=markers[method],
            s=32,
            color=colors[method],
            zorder=4,
            label=f"{short[method]}: {ASSET_NAMES[asset_i]}",
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ASSET_NAMES)
    ax.set_xlim(-0.02, 1.52)
    ax.set_ylim(-0.55, 3.85)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Asset level")
    ax.set_title(f"(b)   Enterprise {case['id']}", loc="left", fontsize=9, color="#1C2B39", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(1.25, 3.62, "Funded by", ha="center", va="center", fontsize=6.5, color="#5C6B7A")
    ax.legend(
        frameon=False,
        fontsize=6.2,
        loc="upper center",
        bbox_to_anchor=(0.42, -0.18),
        ncol=2,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def _round(value: float, digits: int = 3) -> float:
    return float(np.round(value, digits))


def run() -> dict:
    rng = np.random.default_rng(SEED)
    population = simulate(rng)
    raw = np.column_stack([population.indicators, population.distractors])
    knowledge_groups = tuple(np.array([i]) for i in range(4))
    raw_groups = tuple(np.arange(sl.start, sl.stop) for sl in population.asset_slices)
    oracle_lifts = true_lifts(population.latents, DELTA)
    oracle = oracle_lifts.argmax(axis=1)
    ceiling = float(oracle_lifts.max(axis=1).mean())

    kaid_choice, kaid_pred = oof_kaid(population.scores, population.audit, DELTA)
    raw_choice, raw_pred = oof_action(raw, population.audit, raw_groups, DELTA)
    del knowledge_groups
    choices = {
        "Lowest score": population.scores.argmin(axis=1),
        "Raw gradient boosting": raw_choice,
        "Equal-weight knowledge rules": rule_action(population.scores, DELTA),
        "KAID": kaid_choice,
    }
    lifts = {
        name: realized_lift(population.latents, choice, DELTA) for name, choice in choices.items()
    }
    hits = {name: choices[name] == oracle for name in METHOD_ORDER}

    summary = []
    for name in METHOD_ORDER:
        mean_lift, interval = mean_interval(lifts[name], rng)
        summary.append(
            {
                "method": name,
                "hit_rate": _round(float(hits[name].mean())),
                "mean_lift": _round(mean_lift),
                "lift_ci": [_round(interval[0]), _round(interval[1])],
            }
        )

    comparisons = []
    raw_p = []
    for name in METHOD_ORDER[:-1]:
        raw_p.append(wilcoxon_p(lifts["KAID"], lifts[name]))
    adjusted = holm(raw_p)
    for name, p_value, holm_p in zip(METHOD_ORDER[:-1], raw_p, adjusted):
        difference = float((lifts["KAID"] - lifts[name]).mean())
        comparisons.append(
            {
                "comparison": f"KAID vs {name}",
                "kaid_mean_lift": _round(float(lifts["KAID"].mean())),
                "baseline_mean_lift": _round(float(lifts[name].mean())),
                "mean_difference": _round(difference),
                "difference_ci": [_round(v) for v in paired_diff_interval(lifts["KAID"], lifts[name], rng)],
                "wilcoxon_p": p_value,
                "holm_p": holm_p,
                "hit_rate_difference": _round(float(hits["KAID"].mean() - hits[name].mean())),
                "mcnemar_p": mcnemar_p(hits["KAID"], hits[name]),
            }
        )

    sensitivity = []
    for delta in SENSITIVITY:
        if delta == DELTA:
            sens_choices = choices
            sens_lifts = lifts
        else:
            sens_kaid, _ = oof_kaid(population.scores, population.audit, delta)
            sens_raw, _ = oof_action(raw, population.audit, raw_groups, delta)
            sens_choices = {
                "Lowest score": population.scores.argmin(axis=1),
                "Raw gradient boosting": sens_raw,
                "Equal-weight knowledge rules": rule_action(population.scores, delta),
                "KAID": sens_kaid,
            }
            sens_lifts = {
                name: realized_lift(population.latents, choice, delta)
                for name, choice in sens_choices.items()
            }
        sensitivity.append(
            {
                "delta": delta,
                "mean_lift": {name: _round(float(sens_lifts[name].mean())) for name in METHOD_ORDER},
                "hit_rate": {
                    name: _round(float((sens_choices[name] == true_lifts(population.latents, delta).argmax(1)).mean()))
                    for name in METHOD_ORDER
                },
            }
        )

    case_id = select_case(lifts, choices, oracle)
    case = {
        "id": case_id,
        "observed": [_round(float(v)) for v in population.scores[case_id]],
        "latent": [_round(float(v)) for v in population.latents[case_id]],
        "oracle": int(oracle[case_id]),
        "oracle_name": ASSET_NAMES[int(oracle[case_id])],
        "choices": {name: int(choices[name][case_id]) for name in METHOD_ORDER},
        "choice_names": {name: ASSET_NAMES[int(choices[name][case_id])] for name in METHOD_ORDER},
        "lifts": {name: _round(float(lifts[name][case_id])) for name in METHOD_ORDER},
    }

    plot_framework("Figure_1.png")
    plot_results("Figure_2.png", summary, ceiling, case)

    payload = {
        "seed": SEED,
        "n_enterprises": N_ENTERPRISES,
        "n_folds": N_FOLDS,
        "delta": DELTA,
        "pair_weights": {"know_how_x_know_who": PAIR_MAIN, "know_what_x_know_why": PAIR_SECOND},
        "indicator_noise_sd": INDICATOR_NOISE,
        "audit_noise_sd": AUDIT_NOISE,
        "ceiling_mean_lift": _round(ceiling),
        "summary": summary,
        "comparisons": comparisons,
        "sensitivity": sensitivity,
        "case": case,
        "audit_mae": {
            "KAID": _round(float(np.mean(np.abs(kaid_pred - population.audit)))),
            "Raw gradient boosting": _round(float(np.mean(np.abs(raw_pred - population.audit)))),
        },
        "versions": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "scikit-learn": __import__("sklearn").__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with open("results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    _check(payload)
    print(json.dumps(payload, indent=2))
    return payload


def _check(payload: dict) -> None:
    by_name = {row["method"]: row for row in payload["summary"]}
    kaid = by_name["KAID"]["mean_lift"]
    rules = by_name["Equal-weight knowledge rules"]["mean_lift"]
    raw = by_name["Raw gradient boosting"]["mean_lift"]
    gap = by_name["Lowest score"]["mean_lift"]
    if not (kaid > rules > gap and kaid > raw):
        raise RuntimeError(
            "Expected KAID to beat the knowledge rules, raw boosting, and the lowest-score rule. "
            f"Got KAID {kaid}, rules {rules}, raw {raw}, lowest {gap}."
        )
    holm_values = [row["holm_p"] for row in payload["comparisons"]]
    if max(holm_values) >= 0.05:
        raise RuntimeError(f"A Holm-adjusted comparison is not below 0.05: {holm_values}")
    for name in ("Figure_1.png", "Figure_2.png"):
        with open(name, "rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError(f"{name} is not a PNG.")


if __name__ == "__main__":
    run()
