#!/usr/bin/env bash
# Grid search over the combined-loss weights (alpha: focal, beta: pixel accuracy, dice: 1 - alpha - beta)
# at 2 m resolution on 50% of the tiles. Prints the validation accuracy at the best validation loss.
#
# Author: Marco L. Della Vedova <marco.dellavedova@chalmers.se>
set -euo pipefail

for weights in "0 0" "1 0" "0 1" "0.2 0.4" "0.2 0.6" "0.25 0.5" "0.3 0.4" "0.4 0.4"; do
    set -- $weights
    python scripts/train.py --resolution 2 --alpha "$1" --beta "$2" \
        --batch-size 8 --lr 0.001 --epochs 30 --patience 8 --fraction 0.5 \
        --out "runs/loss_grid/alpha$1_beta$2"
done

python - <<'EOF'
import csv, glob
for log in sorted(glob.glob("runs/loss_grid/*/log.csv")):
    best = min(csv.DictReader(open(log)), key=lambda r: float(r["val_loss"]))
    print(f"{log.split('/')[2]:24s} val_accuracy={float(best['val_accuracy']):.4f}")
EOF
