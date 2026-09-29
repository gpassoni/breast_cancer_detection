# Data

No data is stored in this repository. Scripts read from `$ROOT_DIR/data/` (set `ROOT_DIR` in `.env`, default: repository root).

## DMR-IR (public) — classification

The Database for Mastology Research with Infrared Image is available for research use from
the [Visual Lab, Universidade Federal Fluminense](https://visual.ic.uff.br/dmi/).
The experiments use 1,000 frontal frames (500 healthy, 500 cancer) from 44 patients, with the
matching breast-region annotations:

```
data/dmrir/lab_database/
├── database/{normal,abnormal}/   # thermograms (224x224)
└── labels/{normal,abnormal}/     # breast-region masks, same file names
```

All splits are grouped by patient. Patient IDs are parsed from the file names of the three
DMR-IR export formats (`198-1.txt.jpg` → 198, `TFRON_V185_27-8-2013_0.jpg` → 185,
`IR_2015-04-17_0007.jpg` → acquisition session 2015-04-17); see
`thermal_bc.data.dmrir.patient_id_from_path`.

## Private clinical thermograms — segmentation and SSL pretraining

The FLIR images used for segmentation and self-supervised pretraining (one image per patient)
were collected under a clinical data agreement and cannot be redistributed. Raw radiometric
FLIR JPEGs can be converted to temperature maps with
`thermal_bc.data.io.convert_raw_flir_to_numpy`. The code expects:

```
data/
├── images/              # thermal images (.png)
├── masks/MD/, masks/ME/ # right / left breast masks (.tiff)
├── test Flir/           # external test set: processed/*.png, labels/{left,right}_breast/*.tiff
└── ssl_images/ssl_images_npy/   # unlabeled images for VICReg pretraining
```

Then run:

```bash
python scripts/segmentation/prepare_masks.py          # -> data/mask_merged, data/images_numpy, data/mask_preprocessed
python scripts/segmentation/prepare_flir_test_set.py  # -> data/test_flir/{images,masks}
```

To use your own thermal dataset, provide images and binary breast masks in the same layout.

## Trained weights

Not distributed (trained on the private data above). Scripts expect them in `weights/`:

```
weights/
├── segmentation_r2attunet.pth    # R2AttU-Net state dict
├── segmentation_r2attunet.json   # its hyperparameters (same as configs/segmentation_best.json)
├── vicreg_resnet50.pth           # VICReg ResNet-50 backbone state dict
└── vicreg_adapted/fold{0-4}.pth  # backbone after in-domain adaptation, one per CV fold
```
