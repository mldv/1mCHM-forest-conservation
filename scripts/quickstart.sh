#!/usr/bin/env bash
# Quick end-to-end run that fits on a laptop (CPU is enough): build the tiles of a 10x10 km area
# (32 training, 9 validation, 11 test tiles, about half of the labelled pixels of high value in each split;
# about 120 MB of CHM and 190 MB of CHM metadata are downloaded), then train and test the 20 m and 10 m models.
# Everything is written to quickstart/.
# With so few tiles the models are trained with a small batch size and few epochs:
# the results check that the pipeline works, they are not comparable with the paper.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=quickstart
BBOX="695040 6636800 705280 6647040"
EPOCHS=${EPOCHS:-15}
PYTHON=${PYTHON:-python}
export CACHE=${CACHE:-$PWD/$OUT/cache}

$PYTHON scripts/make_tiles.py --bbox $BBOX --out $OUT/tiles

for res in 20 10; do
    for split in training validation test; do
        $PYTHON scripts/build_h5.py $split --resolution $res --tiles $OUT/tiles --out $OUT/h5 --workers 2
    done
    $PYTHON scripts/train.py --resolution $res --h5 $OUT/h5 --out $OUT/runs/${res}m \
        --batch-size 4 --lr 0.001 --epochs "$EPOCHS" --workers 0
done

$PYTHON scripts/evaluate.py --resolutions 20 10 --runs $OUT/runs --h5 $OUT/h5 --out $OUT/results.csv
