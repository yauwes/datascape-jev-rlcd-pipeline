# Generate a fine-tune kernel dir from kernels/ft/ft.py (early stopping, resumable, checkpointed weights + preds + history).
# python work/mk_ft.py NAME ACCOUNT MODEL IMG LR LAYER_DECAY BS TAG [extra_json]
import sys, json, os
name, acct, model, img, lr, ld, bs, tag = sys.argv[1:9]; extra = json.loads(sys.argv[9]) if len(sys.argv) > 9 else {}
d = f'kernels/{name}'; os.makedirs(d, exist_ok=True)
cfg = dict(model=model, img=int(img), lr=float(lr), layer_decay=float(ld), bs=int(bs), tag=tag, **extra)
s = open('kernels/ft/ft.py').read().replace("CFG.update(json.loads(os.environ.get('CFG', '{}')))", f"CFG.update({cfg!r})\nCFG.update(json.loads(os.environ.get('CFG', '{{}}')))")
open(f'{d}/ft.py', 'w').write(s)
json.dump({"id": f"{acct}/datascape-{name}", "title": f"datascape-{name}", "code_file": "ft.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": [f"{acct}/datascape-small-512", f"{acct}/datascape-aux"], "competition_sources": [], "kernel_sources": []}, open(f'{d}/kernel-metadata.json', 'w'), indent=1)
print('wrote', d, cfg)
