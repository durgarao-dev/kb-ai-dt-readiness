# Knowledge-based AI decision support for digital transformation readiness

Short communication and the one script that produces its evidence.

**[Read the paper](PAPER.md)**

KAID turns enterprise innovation indicators into two complementarity rules, learns the weight of each rule from a noisy readiness audit, and funds one knowledge asset.

The 360 enterprises are simulated from the process disclosed in the paper. On that benchmark KAID reaches a mean readiness lift of 0.367, against a best attainable lift of 0.372. Equal-weight rules reach 0.257. Raw gradient boosting reaches 0.336. Funding the lowest score reaches 0.195.

## Run

```bash
python -m pip install -r requirements.txt
python kb_dt_model.py
```

Seed `20261024`. The command rewrites `Figure_1.png`, `Figure_2.png`, and `results.json`.

## Repository

| File | Role |
| --- | --- |
| `PAPER.md` | Short communication, two figures, two tables |
| `kb_dt_model.py` | Population, models, decision rules, figures, checks |
| `results.json` | Metrics from the seeded run reported in the paper |
| `requirements.txt` | NumPy, SciPy, scikit-learn, Matplotlib |
| `Figure_1.png` | Decision-support framework |
| `Figure_2.png` | Readiness lift and illustrative enterprise |
| `highlights.md` | Separate highlights file for submission |

Code is released under the MIT license. The manuscript text is the author's.
