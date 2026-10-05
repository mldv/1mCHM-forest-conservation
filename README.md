# Meter-resolution canopy height data alone can be used for predicting the conservation value of Boreal forests

Code and data for the paper by Marco L. Della Vedova and Mattias Wahde, *IEEE Geoscience and Remote Sensing Letters* (2025).

A U-Net takes a canopy height model (CHM) tile as its only input and predicts, for every pixel, the probability that the forest has high conservation value. Models are trained on 1280×1280 m tiles at resolutions from 40 m down to 1 m. At 1 m the model reaches a test accuracy above 0.95 in the South Boreal region of Sweden.

## Repository layout

```
data/
  labels.gpkg.zip   ground-truth polygons, dissolved per class (naturalness 0 = low, 1 = high), EPSG:3006
  tiles.gpkg        1280 m tile grid with the training / validation / test split
src/chmcons/        model, data handling, losses, metrics
scripts/
  make_tiles.py     CHM + labels -> 2-band GeoTIFF tiles
  build_h5.py       tiles -> HDF5 at a given resolution
  train.py          train one model
  evaluate.py       test-set metrics (Table 1)
  fig_resolution.py Fig. 3
  fig_examples.py   Fig. 2
  loss_grid.sh      grid search over the loss weights
  predict.py        prediction map for any bounding box in Sweden
```

## Installation

You need Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The CHM ([Skogsstyrelsen](https://www.skogsstyrelsen.se/sjalvservice/karttjanster/skogliga-grunddata), *trädhöjd* layer) is downloaded on demand from the agency's public FTP by [SkogDataPy](https://github.com/mldv/SkogDataPy). Downloaded files are cached in the folder given by the `CACHE` environment variable, or by a `.env` file in the working directory:

```bash
echo "CACHE=/path/to/chm/cache" > .env
```

The full study area covers a large part of southern Sweden, so plan disk space for the cache, for the tiles and for the HDF5 files.

## Reproducing the results

Run every command from the repository root.

**1. Tiles.** Downloads the CHM, rasterizes the labels and writes `tiles/{training,validation,test}/*.tif`. Grid cells without CHM coverage or without labels are skipped. Tiles that already exist are not rebuilt, so the script can be resumed.

```bash
python scripts/make_tiles.py
```

**2. HDF5 datasets.** Writes one file per split and resolution, e.g. `h5/training-1m.h5`. The CHM is block-averaged and the labels are majority-voted.

```bash
for res in 1 2 5 8 10 20 40; do
  for split in training validation test; do
    python scripts/build_h5.py $split --resolution $res
  done
done
```

**3. Training.** Writes `runs/<res>m/model.pth` (the epoch with the lowest validation loss) and `runs/<res>m/log.csv`.

```bash
python scripts/train.py --resolution 1
```

All models use the combined loss with α = 0.3 and β = 0.4, Adam, at most 50 epochs, and early stopping with patience 10. Batch size and learning rate depend on the resolution and default to the values used for the published models:

| Resolution | Tile (px) | Batch size | Learning rate |
|-----------:|----------:|-----------:|--------------:|
| 40 m | 32   | 32   | 0.001 |
| 20 m | 64   | 1024 | 0.016 |
| 10 m | 128  | 1024 | 0.016 |
| 8 m  | 160  | 1024 | 0.016 |
| 5 m  | 256  | 1024 | 0.016 |
| 2 m  | 640  | 6    | 0.005 |
| 1 m  | 1280 | 2    | 0.001 |

If more than one GPU is visible, `torch.nn.DataParallel` is used.

**4. Evaluation (Table 1).** Writes `results/results.csv`. Metrics are computed over all labelled (non-background) test pixels, with predictions thresholded at 0.5.

```bash
python scripts/evaluate.py
```

**5. Figures.**

```bash
python scripts/fig_resolution.py                                   # Fig. 3
python scripts/fig_examples.py tiles/validation/636160_6689280_1280.tif   # Fig. 2, top row (validation tile)
python scripts/fig_examples.py tiles/validation/506880_6268160_1280.tif   # Fig. 2, bottom row (validation tile)
```

**Loss-function grid search.** Trains 2 m models on half of the tiles for each (α, β) pair, then prints the validation accuracy at the best validation loss:

```bash
bash scripts/loss_grid.sh
```

## Prediction for any area

Coordinates are SWEREF99 TM (EPSG:3006). The output is a GeoTIFF of per-pixel probabilities of high conservation value:

```bash
python scripts/predict.py --bbox 688640 6580480 691000 6582000 --checkpoint runs/1m/model.pth --png
```

The area is processed in independent 1280×1280 m windows, so artifacts can appear at window borders and in the zero-padded windows at the edges.

## Data

`data/labels.gpkg.zip` contains two geometries:
- **High conservation value (1):** the union of habitat-classed Natura 2000 areas and woodland key habitats, inventoried by forestry companies and by the Swedish Forest Agency.
- **Low conservation value (0):** the union of forests harvested 1990–2000, stands aged 30–80 years, and older stands assessed as having low natural value.

Areas labelled both high and low are treated as background when the tiles are made. See the paper and [Della Vedova & Wahde (2024)](https://doi.org/10.1080/22797254.2024.2441834) for details on the label sources. The tile grid and split follow the same geographical stratification as that earlier work.

## Citation

```bibtex
@article{dellavedova2025meter,
  title   = {Meter-resolution canopy height data alone can be used for predicting the conservation value of Boreal forests},
  author  = {Della Vedova, Marco L. and Wahde, Mattias},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2025}
}
```

## Acknowledgements

This research was partially supported by VINNOVA, project 2022-01702. The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725.

## License

GPL-3.0. See [LICENSE](LICENSE).
