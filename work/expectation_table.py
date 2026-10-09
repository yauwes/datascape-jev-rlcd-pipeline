# Expected vs actual public LB for every team submission -> work/expectation_vs_lb.csv (read by the notebook, section 12).
# Needs the Kaggle CLI logged in as a team member (env -u KAGGLE_API_TOKEN). Usage: python work/expectation_table.py
import json, subprocess, numpy as np, pandas as pd
L = pd.read_csv('submissions/ledger.csv')
subs = json.loads(subprocess.run(['kaggle', 'competitions', 'submissions', 'datascape-2026-data-mining-competition', '--format', 'json'], capture_output=True, text=True).stdout)
first = lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
rows = []
for s in subs:
    n = s['fileName'].rsplit('.', 1)[0]; lb = float(s['publicScore']) if s['publicScore'] else np.nan; led = L[(L.name == n) | (L.submission_id == n)]
    cv = first(led.oof_macro_f1) if len(led) else np.nan
    if np.isnan(cv) and len(led): cv = first(led.cv_macro_f1)
    ex = first(led.expected_lb) if len(led) else np.nan
    rows.append(dict(date=s['date'][:10], sub=n[:34], cv_or_oof=cv, expected_lb=ex, public_lb=lb, gap=round(lb - ex, 4) if not np.isnan(ex) else np.nan))
D = pd.DataFrame(rows).sort_values('date'); D.to_csv('work/expectation_vs_lb.csv', index=False); print(D.tail(4).to_string(index=False))
