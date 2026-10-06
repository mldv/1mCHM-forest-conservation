#!/usr/bin/env bash
# Quick end-to-end run that fits on a laptop (CPU is enough): build the tiles of a 10x10 km area
# (32 training, 9 validation, 11 test tiles, about half of the labelled pixels of high value in each split;
# about 120 MB of CHM and 190 MB of CHM metadata are downloaded), then train and test the 20 m and 10 m models.
# Everything is written to quickstart/.
# With so few tiles the models are trained with a small batch size, few epochs and a fixed seed:
# the results check that the pipeline works, they are not comparable with the paper.
#
# Author: Marco L. Della Vedova <marco.dellavedova@chalmers.se>
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=quickstart
BBOX="695040 6636800 705280 6647040"
RESOLUTIONS="20 10"
EPOCHS=${EPOCHS:-15}
PYTHON=${PYTHON:-python}
export CACHE=${CACHE:-$PWD/$OUT/cache}
export PYTHONWARNINGS="ignore:Can't initialize NVML"   # harmless PyTorch warning on machines without a working NVIDIA driver

step() { printf '\n=== %s\n' "$*"; }

step "1/4  Tiles of a 10x10 km area (bbox $BBOX) -> $OUT/tiles"
echo "     CHM is read from $CACHE and downloaded from Skogsstyrelsen if missing"
$PYTHON scripts/make_tiles.py --bbox $BBOX --out $OUT/tiles

step "2/4  HDF5 files at $RESOLUTIONS m -> $OUT/h5"
for res in $RESOLUTIONS; do
    for split in training validation test; do
        $PYTHON scripts/build_h5.py $split --resolution $res --tiles $OUT/tiles --out $OUT/h5 --workers 2
    done
done

for res in $RESOLUTIONS; do
    step "3/4  Training the $res m model ($EPOCHS epochs) -> $OUT/runs/${res}m"
    echo "     per epoch: loss train/validation, accuracy train/validation"
    $PYTHON scripts/train.py --resolution $res --h5 $OUT/h5 --out $OUT/runs/${res}m \
        --batch-size 2 --lr 0.001 --epochs "$EPOCHS" --seed 0 --workers 0
done

step "4/4  Test set evaluation -> $OUT/results.csv"
$PYTHON scripts/evaluate.py --resolutions $RESOLUTIONS --runs $OUT/runs --h5 $OUT/h5 --out $OUT/results.csv

step "Done. Results in $OUT/results.csv, models in $OUT/runs/<resolution>m/model.pth"
