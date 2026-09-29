# Data

No data is stored in this repository. Scripts read from `$ROOT_DIR/data/` (set `ROOT_DIR` in `.env`, default: repository root).

## DMR-IR (public) — classification

The Database for Mastology Research with Infrared Image is available for research use from
the [Visual Lab, Universidade Federal Fluminense](https://visual.ic.uff.br/dmi/).
Arrange the frontal thermograms by diagnosis:

```
data/dmrir/lab_database/database/
├── normal/      # healthy
└── abnormal/    # cancer
```

Train/validation/test splits are grouped by patient. The patient ID is parsed from each file
name with `--patient-id-pattern` (default `PAC_(\d+)_`, e.g. `PAC_17_023.jpg` → patient 17);
pass a different regex if your copy of the dataset uses another naming scheme.

## Private clinical thermograms — segmentation and SSL pretraining

The FLIR images used for segmentation and self-supervised pretraining were collected under a
clinical data agreement and cannot be redistributed. The code expects:

```
data/
├── images/              # thermal images (.png)
├── masks/MD/, masks/ME/ # right / left breast masks (.tiff)
├── test Flir/           # external test set: processed/*.png, labels/{left,right}_breast/*.tiff
└── ssl_images/ssl_images_npy/   # unlabeled images for VICReg pretraining
```

Then run:

```bash
python scripts/prepare_masks.py          # -> data/mask_merged, data/images_numpy, data/mask_preprocessed
python scripts/prepare_flir_test_set.py  # -> data/test_flir/{images,masks}
```

To use your own thermal dataset, provide images and binary breast masks in the same layout.
