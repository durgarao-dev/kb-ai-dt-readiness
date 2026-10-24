"""Knowledge-based AI decision support for digital transformation readiness.

Reproducible validation for the short communication in PAPER.md.
Enterprises are simulated from a fully disclosed process. Re-run:

    python kb_dt_model.py

The command rewrites fig1_architecture.png, fig2_performance.png, and results.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261024
N_ENTERPRISES = 2000
N_FOLDS = 5
N_BOOT = 2000
BENCHMARK = 0.60
TARGET_PREVALENCE = 0.40
INDICATOR_NOISE = 0.12
# Noise on the readiness index. Small enough that readiness is learnable,
# large enough that the label is not a deterministic function of knowledge.
INDEX_NOISE = 0.18

ASSETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Know-how", ("Training intensity", "Process innovation", "R&D intensity")),
    ("Know-who", ("Customer cooperation", "Supplier cooperation", "University cooperation")),
    ("Know-what", ("Product innovation", "Innovation sales share")),
    ("Know-why", ("Scientific sourcing", "Research-project share")),
)
DISTRACTORS = ("Plant age", "Fleet size", "Unrelated certifications")
ASSET_NAMES = [name for name, _ in ASSETS]


@dataclass(frozen=True)
class Population:
    indicators: np.ndarray
    distractors: np.ndarray
    scores: np.ndarray
    latents: np.ndarray
    ready: np.ndarray
    asset_slices: tuple[slice, ...]


def readiness_index(latents: np.ndarray) -> np.ndarray:
    """Readiness from two complementary pairs, not from a sum of assets.

    Know-how creates readiness only together with know-who, and know-what only
    together with know-why. A linear score cannot represent that pattern.
    """
    how, who, what, why = (latents[:, i] for i in range(4))
    return 2.40 * how * who + 1.60 * what * why


def _threshold_for_share(observed: np.ndarray, target: float) -> float:
    lo, hi = float(observed.min()), float(observed.max())
    for _ in range(60):
        mid = (lo + hi) / 2.0
        share_ready = float((observed > mid).mean())
        if share_ready > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def simulate(rng: np.random.Generator) -> Population:
    """Draw enterprises, noisy innovation indicators, and a readiness label.

    Each knowledge latent is shared by its indicators. Three distractors are
    independent of readiness, so a model that uses every raw column can chase
    them. The label is a noisy threshold on the complementary readiness index.
    """
    latents = rng.beta(2.0, 2.0, size=(N_ENTERPRISES, 4))
    columns: list[np.ndarray] = []
    slices: list[slice] = []
    cursor = 0
    for asset_i, (_, items) in enumerate(ASSETS):
        start = cursor
        for _ in items:
            observed = latents[:, asset_i] + rng.normal(0.0, INDICATOR_NOISE, N_ENTERPRISES)
            columns.append(np.clip(observed, 0.0, 1.0))
            cursor += 1
        slices.append(slice(start, cursor))

    indicators = np.column_stack(columns)
    scores = np.column_stack([indicators[:, sl].mean(axis=1) for sl in slices])
    distractors = rng.beta(2.0, 2.0, size=(N_ENTERPRISES, len(DISTRACTORS)))

    index = readiness_index(latents)
    observed = index + rng.normal(0.0, INDEX_NOISE, N_ENTERPRISES)
    threshold = _threshold_for_share(observed, TARGET_PREVALENCE)
    ready = (observed > threshold).astype(int)
    return Population(indicators, distractors, scores, latents, ready, tuple(slices))


def cronbach_alpha(items: np.ndarray) -> float:
    item_count = items.shape[1]
    item_variance = items.var(axis=0, ddof=1).sum()
    total_variance = items.sum(axis=1).var(ddof=1)
    return float((item_count / (item_count - 1.0)) * (1.0 - item_variance / total_variance))


def make_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.06,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=SEED,
    )


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        ]
    )


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (probability >= edges[i]) & (probability < edges[i + 1])
        else:
            mask = (probability >= edges[i]) & (probability <= edges[i + 1])
        if not np.any(mask):
            continue
        error += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(error)


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": expected_calibration_error(y_true, probability),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
    }


def oof_probability(builder, features: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    probability = np.zeros(len(y_true), dtype=float)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in folds.split(features, y_true):
        model = builder()
        model.fit(features[train_idx], y_true[train_idx])
        probability[test_idx] = model.predict_proba(features[test_idx])[:, 1]
    return probability


def counterfactual_lifts(model, scores: np.ndarray, benchmark: float = BENCHMARK) -> np.ndarray:
    """Increase in predicted readiness if each knowledge asset is raised to the bar."""
    baseline = model.predict_proba(scores)[:, 1]
    lifts = np.zeros_like(scores)
    for asset_i in range(scores.shape[1]):
        intervened = scores.copy()
        intervened[:, asset_i] = np.maximum(intervened[:, asset_i], benchmark)
        lifts[:, asset_i] = model.predict_proba(intervened)[:, 1] - baseline
    return lifts


def oracle_binding_asset(latents: np.ndarray, benchmark: float = BENCHMARK) -> np.ndarray:
    """Asset whose lift to the knowledge bar most increases the true readiness index."""
    baseline = readiness_index(latents)
    gains = np.zeros_like(latents)
    for asset_i in range(latents.shape[1]):
        intervened = latents.copy()
        intervened[:, asset_i] = np.maximum(intervened[:, asset_i], benchmark)
        gains[:, asset_i] = readiness_index(intervened) - baseline
    return gains.argmax(axis=1)


def kaid_decisions(scores: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
    """Out-of-fold actions from three rules, plus fold-averaged asset importance."""
    n_obs, n_assets = scores.shape
    kaid_choice = np.zeros(n_obs, dtype=int)
    gap_choice = np.zeros(n_obs, dtype=int)
    importance_choice = np.zeros(n_obs, dtype=int)
    importance_sum = np.zeros(n_assets, dtype=float)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold_id, (train_idx, test_idx) in enumerate(folds.split(scores, y_true)):
        model = make_boosting()
        model.fit(scores[train_idx], y_true[train_idx])
        kaid_choice[test_idx] = counterfactual_lifts(model, scores[test_idx]).argmax(axis=1)

        gap = np.maximum(0.0, BENCHMARK - scores[test_idx])
        gap_choice[test_idx] = gap.argmax(axis=1)

        importance = permutation_importance(
            model,
            scores[train_idx],
            y_true[train_idx],
            n_repeats=15,
            scoring="roc_auc",
            n_jobs=1,
            random_state=SEED + fold_id,
        )
        weights = np.maximum(importance.importances_mean, 0.0)
        importance_sum += weights
        importance_choice[test_idx] = int(weights.argmax())

    return {
        "kaid": kaid_choice,
        "equal_weight_gap": gap_choice,
        "importance_only": importance_choice,
        "importance_mean": importance_sum / N_FOLDS,
    }


def evaluation_mask(latents: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Not-ready enterprises that are still short of the bar on at least one asset."""
    below_bar = (latents < BENCHMARK).any(axis=1)
    return (y_true == 0) & below_bar


def hit_rate(choice: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> float:
    return float((choice[mask] == oracle[mask]).mean())


def bootstrap_metric_cis(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    hits: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, dict]:
    """Percentile intervals that treat out-of-fold scores as fixed."""
    n_obs = len(y_true)
    auc_draws = {name: [] for name in probabilities}
    raw_diff_draws = []
    knowledge_diff_draws = []
    rule_names = [name for name in hits if name not in {"mask", "oracle"}]
    hit_draws = {name: [] for name in rule_names}
    hit_index = np.flatnonzero(hits["mask"])

    for _ in range(N_BOOT):
        sample = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y_true[sample])) < 2:
            continue
        fold_auc = {}
        for name, probability in probabilities.items():
            fold_auc[name] = roc_auc_score(y_true[sample], probability[sample])
            auc_draws[name].append(fold_auc[name])
        raw_diff_draws.append(fold_auc["KAID"] - fold_auc["Raw gradient boosting"])
        knowledge_diff_draws.append(fold_auc["KAID"] - fold_auc["Knowledge logistic"])

        hit_sample = rng.choice(hit_index, size=len(hit_index), replace=True)
        for name in rule_names:
            matched = hits[name][hit_sample] == hits["oracle"][hit_sample]
            hit_draws[name].append(float(matched.mean()))

    def _ci(values: list[float]) -> list[float]:
        array = np.asarray(values, dtype=float)
        return [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))]

    return {
        "auc": {name: _ci(values) for name, values in auc_draws.items()},
        "kaid_minus_raw_boosting": _ci(raw_diff_draws),
        "kaid_minus_knowledge_logistic": _ci(knowledge_diff_draws),
        "hit_rate": {name: _ci(values) for name, values in hit_draws.items()},
    }


def _box(ax, x, y, w, h, face, title, lines, title_size=9.5, body_size=8.0) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=face,
        edgecolor="#1F3A5F",
        linewidth=1.1,
        mutation_aspect=0.6,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color="#1F3A5F")
    ax.text(x + w / 2, y + h / 2 - 0.08, lines, ha="center", va="center", fontsize=body_size, color="#243447", linespacing=1.35)


def _arrow(ax, start, end) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#1F3A5F",
            shrinkA=0,
            shrinkB=0,
        )
    )


def plot_architecture(path: str) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    _box(
        ax, 0.35, 1.15, 3.15, 3.15, "#E8F1FA", "1  ·  Innovation data",
        "Enterprise innovation indicators\n"
        "grouped into four knowledge assets\n\n"
        "Know-how     Know-who\n"
        "Know-what    Know-why\n\n"
        "Asset score = mean of its indicators",
    )
    _box(
        ax, 4.45, 1.15, 3.15, 3.15, "#FFF4E0", "2  ·  AI predictor",
        "Gradient boosting on the four\n"
        "knowledge scores only\n\n"
        "Out-of-fold probability that the\n"
        "enterprise is ready for digital\n"
        "transformation",
    )
    _box(
        ax, 8.55, 1.15, 3.15, 3.15, "#E7F5EF", "3  ·  Decision rule",
        "Raise one asset to the 0.60 bar\n"
        "and recompute the probability\n\n"
        "Recommend the asset with the\n"
        "largest readiness lift\n\n"
        "Bands: lagging · emerging · ready",
    )
    _arrow(ax, (3.55, 2.72), (4.40, 2.72))
    _arrow(ax, (7.65, 2.72), (8.50, 2.72))
    ax.text(
        6.0, 0.48,
        "Distractor indicators are kept in the raw-data baselines and excluded from the knowledge model.",
        ha="center", va="center", fontsize=8.0, color="#526173",
    )
    ax.set_title(
        "Knowledge-based AI decision-support model",
        loc="left", fontsize=13, color="#1F3A5F", fontweight="bold", pad=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_performance(path: str, y_true: np.ndarray, probabilities: dict[str, np.ndarray], decision_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    fig.patch.set_facecolor("white")
    colors = {
        "KAID": "#0072B2",
        "Knowledge logistic": "#009E73",
        "Raw gradient boosting": "#E69F00",
        "Raw logistic": "#CC79A7",
    }
    ax = axes[0]
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, probability)
        auc = roc_auc_score(y_true, probability)
        ax.plot(fpr, tpr, color=colors[name], lw=2.0, label=f"{name}  {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#9AA7B2", lw=1.0, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("a   Out-of-fold discrimination", loc="left", fontsize=11, color="#1F3A5F")
    ax.legend(frameon=False, fontsize=8.0, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    names = [row["label"] for row in decision_rows]
    rates = [row["rate"] for row in decision_rows]
    lows = [row["rate"] - row["ci"][0] for row in decision_rows]
    highs = [row["ci"][1] - row["rate"] for row in decision_rows]
    y_pos = np.arange(len(names))[::-1]
    bar_colors = ["#0072B2", "#009E73", "#E69F00"]
    ax.barh(y_pos, rates, color=bar_colors, height=0.62, xerr=[lows, highs], capsize=3, ecolor="#1F3A5F")
    ax.axvline(0.25, color="#9AA7B2", ls="--", lw=1.0, label="Chance (1/4)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Hit rate on the binding knowledge asset")
    ax.set_title("b   Decision-rule recovery", loc="left", fontsize=11, color="#1F3A5F")
    ax.legend(frameon=False, fontsize=8.0, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _round(value: float, digits: int = 3) -> float:
    return float(np.round(value, digits))


def run() -> dict:
    rng = np.random.default_rng(SEED)
    population = simulate(rng)
    raw = np.column_stack([population.indicators, population.distractors])
    y_true = population.ready

    probabilities = {
        "KAID": oof_probability(make_boosting, population.scores, y_true),
        "Knowledge logistic": oof_probability(make_logistic, population.scores, y_true),
        "Raw gradient boosting": oof_probability(make_boosting, raw, y_true),
        "Raw logistic": oof_probability(make_logistic, raw, y_true),
    }
    metrics = {name: classification_metrics(y_true, probability) for name, probability in probabilities.items()}

    decisions = kaid_decisions(population.scores, y_true)
    oracle = oracle_binding_asset(population.latents)
    mask = evaluation_mask(population.latents, y_true)
    hit_values = {
        "mask": mask,
        "oracle": oracle,
        "KAID counterfactual": decisions["kaid"],
        "Equal-weight gap": decisions["equal_weight_gap"],
        "Importance only": decisions["importance_only"],
    }
    rates = {name: hit_rate(choice, oracle, mask) for name, choice in hit_values.items() if name not in {"mask", "oracle"}}

    cis = bootstrap_metric_cis(y_true, probabilities, hit_values, rng)
    alphas = {
        name: cronbach_alpha(population.indicators[:, sl])
        for (name, _), sl in zip(ASSETS, population.asset_slices)
    }
    bands = {
        "lagging": float((probabilities["KAID"] < 0.35).mean()),
        "emerging": float(((probabilities["KAID"] >= 0.35) & (probabilities["KAID"] <= 0.65)).mean()),
        "ready": float((probabilities["KAID"] > 0.65).mean()),
    }

    decision_rows = [
        {
            "label": "KAID counterfactual",
            "rate": rates["KAID counterfactual"],
            "ci": cis["hit_rate"]["KAID counterfactual"],
        },
        {
            "label": "Equal-weight gap",
            "rate": rates["Equal-weight gap"],
            "ci": cis["hit_rate"]["Equal-weight gap"],
        },
        {
            "label": "Importance only",
            "rate": rates["Importance only"],
            "ci": cis["hit_rate"]["Importance only"],
        },
    ]

    plot_architecture("fig1_architecture.png")
    plot_performance("fig2_performance.png", y_true, probabilities, decision_rows)

    payload = {
        "seed": SEED,
        "n_enterprises": N_ENTERPRISES,
        "n_folds": N_FOLDS,
        "n_bootstrap": N_BOOT,
        "benchmark": BENCHMARK,
        "prevalence": _round(float(y_true.mean())),
        "ceiling_auc": _round(float(roc_auc_score(y_true, readiness_index(population.latents)))),
        "dgp": {
            "latent": "Beta(2, 2)",
            "index": "2.40*know_how*know_who + 1.60*know_what*know_why",
            "index_noise_sd": INDEX_NOISE,
            "indicator_noise_sd": INDICATOR_NOISE,
        },
        "n_decision_cases": int(mask.sum()),
        "cronbach_alpha": {name: _round(value) for name, value in alphas.items()},
        "metrics": {
            name: {key: _round(value) for key, value in row.items()}
            for name, row in metrics.items()
        },
        "auc_ci": {name: [_round(v) for v in bounds] for name, bounds in cis["auc"].items()},
        "kaid_minus_raw_boosting_auc": [_round(v) for v in cis["kaid_minus_raw_boosting"]],
        "kaid_minus_knowledge_logistic_auc": [_round(v) for v in cis["kaid_minus_knowledge_logistic"]],
        "hit_rate": {name: _round(rate) for name, rate in rates.items()},
        "hit_rate_ci": {name: [_round(v) for v in bounds] for name, bounds in cis["hit_rate"].items()},
        "importance_mean": {
            name: _round(float(value))
            for name, value in zip(ASSET_NAMES, decisions["importance_mean"])
        },
        "readiness_bands": {name: _round(share) for name, share in bands.items()},
        "versions": {
            "numpy": np.__version__,
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
    kaid_auc = payload["metrics"]["KAID"]["roc_auc"]
    logistic_auc = payload["metrics"]["Knowledge logistic"]["roc_auc"]
    raw_auc = payload["metrics"]["Raw gradient boosting"]["roc_auc"]
    kaid_hit = payload["hit_rate"]["KAID counterfactual"]
    gap_hit = payload["hit_rate"]["Equal-weight gap"]
    if kaid_auc < 0.70:
        raise RuntimeError(f"KAID discrimination is too weak to carry the note: {kaid_auc}")
    if kaid_auc <= logistic_auc:
        raise RuntimeError(
            f"Boosting does not beat the linear knowledge model ({kaid_auc:.3f} vs {logistic_auc:.3f})."
        )
    if kaid_auc <= raw_auc:
        raise RuntimeError(
            f"Knowledge scores do not beat raw boosting ({kaid_auc:.3f} vs {raw_auc:.3f})."
        )
    if kaid_hit <= gap_hit:
        raise RuntimeError(
            "KAID counterfactual does not beat the equal-weight gap rule "
            f"({kaid_hit:.3f} vs {gap_hit:.3f})."
        )
    for name in ("fig1_architecture.png", "fig2_performance.png"):
        with open(name, "rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError(f"{name} is not a PNG.")


if __name__ == "__main__":
    run()
