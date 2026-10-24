# Knowledge-based AI decision support for digital transformation readiness

Short communication and the one script that produces its evidence.

**[Read the paper](PAPER.md)**

The model groups enterprise innovation indicators into four knowledge assets (know-how, know-who, know-what, know-why), estimates digital-transformation readiness with gradient boosting, and recommends the asset that most raises that probability when lifted to 0.60.

The enterprises are simulated from the process disclosed in the paper. The script is the validation, not a fit to a private field sample. On 2,000 enterprises the linear knowledge score already screens readiness (ROC-AUC 0.919). The boosted model is justified by the action: it names the binding asset for 74.7% of enterprises that are not ready, against 68.8% for the largest knowledge gap.

## Run

```bash
python -m pip install -r requirements.txt
python kb_dt_model.py
```

Seed `20261024`. The command rewrites `fig1_architecture.png`, `fig2_performance.png`, and `results.json`.

## Repository

| File | Role |
| --- | --- |
| `PAPER.md` | Short communication, two figures, two tables |
| `kb_dt_model.py` | Population, models, decision rules, figures, checks |
| `results.json` | Metrics from the seeded run reported in the paper |
| `requirements.txt` | NumPy, scikit-learn, Matplotlib |
| `fig1_architecture.png` | Model diagram |
| `fig2_performance.png` | ROC curves and decision-rule hit rates |

Code is released under the MIT license. The manuscript text is the author's.
