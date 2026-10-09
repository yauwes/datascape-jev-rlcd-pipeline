# Submission ledger: every candidate = ready CSV + strategy + expected LB; actual LB recorded after submit -> expectation vs reality.
# python work/subreg.py add NAME CSV "strategy" OOF EXP LO HI   |   submit NAME   |   sync   |   show
import sys, os, shutil, json, subprocess, time, datetime, pandas as pd
LED = 'submissions/ledger.csv'; COMP = 'datascape-2026-data-mining-competition'
COLS = ['name', 'created_utc', 'csv', 'strategy', 'oof_macro_f1', 'expected_lb', 'exp_lo', 'exp_hi', 'pred_mix', 'kaggle_ref', 'submitted_utc', 'actual_lb', 'delta_vs_expected', 'within_range']

def load():
    return pd.read_csv(LED) if os.path.exists(LED) else pd.DataFrame(columns=COLS)

def add(name, csv, strategy, oof, exp, lo, hi):
    os.makedirs('submissions', exist_ok=True); d = load(); assert name not in set(d.name), f'{name} exists'
    s = pd.read_csv(csv); ss = pd.read_csv('datascape/sample_submission.csv')
    assert list(s.columns) == ['filename', 'label'] and len(s) == len(ss) and set(s.filename) == set(ss.filename)
    assert set(s.label) <= {'Aman', 'Rusak', 'Lainnya'}
    dst = f'submissions/{name}.csv'; shutil.copy(csv, dst)
    mix = json.dumps(s.label.value_counts(normalize=True).round(3).to_dict())
    row = dict(name=name, created_utc=datetime.datetime.utcnow().isoformat(timespec='minutes'), csv=dst, strategy=strategy, oof_macro_f1=float(oof),
               expected_lb=float(exp), exp_lo=float(lo), exp_hi=float(hi), pred_mix=mix)
    pd.concat([d, pd.DataFrame([row])])[list(dict.fromkeys(list(d.columns) + COLS))].to_csv(LED, index=False); print('registered', name, mix)   # keeps other agents' extra columns

def whoami():
    out = subprocess.run(['kaggle', 'config', 'view'], capture_output=True, text=True).stdout
    return next((l.split(':', 1)[1].strip() for l in out.splitlines() if l.strip().startswith('- username')), '')

def submit(name):
    d = load(); r = d[d.name == name].iloc[0]; assert pd.isna(r.kaggle_ref), 'already submitted'
    assert 'KAGGLE_API_TOKEN' not in os.environ and whoami() == 'ajikajadeh', 'refusing: submit only from ajikajadeh'
    msg = f"{name} | {r.strategy} | OOF {r.oof_macro_f1:.4f} | expected LB {r.expected_lb:.3f} [{r.exp_lo:.3f},{r.exp_hi:.3f}]"[:500]
    subprocess.run(['kaggle', 'competitions', 'submit', COMP, '-f', r.csv, '-m', msg], check=True)
    d.loc[d.name == name, 'submitted_utc'] = datetime.datetime.utcnow().isoformat(timespec='minutes'); d.to_csv(LED, index=False)
    for _ in range(60):
        time.sleep(15)
        if sync(quiet=True): break
    show()

def sync(quiet=False):
    """match submitted rows to Kaggle submissions by description prefix and fill actual LB."""
    d = load(); subs = json.loads(subprocess.run(['kaggle', 'competitions', 'submissions', COMP, '--format', 'json'], capture_output=True, text=True).stdout)
    done = True
    for i, r in d[d.submitted_utc.notna() & d.actual_lb.isna()].iterrows():
        hit = [x for x in subs if str(x.get('description', '')).startswith(r['name'] + ' |')]
        if hit and hit[0].get('publicScore') not in (None, ''):
            lb = float(hit[0]['publicScore']); d.loc[i, ['kaggle_ref', 'actual_lb', 'delta_vs_expected', 'within_range']] = [hit[0]['ref'], lb, round(lb - r.expected_lb, 5), r.exp_lo <= lb <= r.exp_hi]
        else: done = False
    d.to_csv(LED, index=False); return done

def show():
    d = load(); print(d[['name', 'oof_macro_f1', 'expected_lb', 'exp_lo', 'exp_hi', 'actual_lb', 'delta_vs_expected', 'within_range', 'pred_mix']].to_string(index=False))

if __name__ == '__main__':
    a = sys.argv[1:]
    {'add': lambda: add(*a[1:]), 'submit': lambda: submit(a[1]), 'sync': lambda: (sync(), show()), 'show': show}[a[0]]()
