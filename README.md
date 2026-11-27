# WasteAI — Six-class Waste Image Classifier

Educational image-classification app using TrashNet (2,527 images; 6 categories) and transfer learning from ImageNet-pretrained MobileNetV3-Small. Trains a new 6-class classifier head, evaluates on a held-out stratified test partition, and serves predictions with Flask. The ImageNet backbone is pretrained; do not describe this as CNN trained entirely from scratch.

## Mac / Linux Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/Ebuka24Stephen/trashnet-dataset.git raw_dataset
python3 train.py
python3 app.py
```

Open http://127.0.0.1:5000. For macOS MPS, training uses GPU when available. Training requires internet once to download pretrained model weights.

## Files
- `train.py` dataset integrity checks, exact duplicate removal, stratified train/validation/test split (70/15/15), training, model selection, final evaluation, graphs.
- `app.py` Flask inference API.
- `templates/index.html` upload + confidence bars.
- `results/metrics.json` **actual, measured** final result. Do not fabricate test accuracy.
- `results/training_history.csv` per-epoch learning curves.
- `results/split_manifest.json` reproducible split record.
- `results/accuracy.png` and `results/confusion_matrix.png` for research figures.
- `saved_models/waste_model.pth` trained checkpoint.

## Methodological notes
- Dataset: TrashNet resized copy. Original dataset: https://github.com/garythung/trashnet
- Citation: Thung, G. and Yang, M., *Classification of Trash for Recyclability Status*, Stanford CS229 project report (2016), https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf
- Single stratified 70/15/15 split with random seed 42; exact file duplicates removed before splitting. Near-duplicate images are not automatically removed.
- Best checkpoint chosen by validation loss; final test evaluation once.
- Limitations: class imbalance, white-background dataset, domain shift, and probability calibration.
- Dataset usage/attribution and image licensing should be reviewed before republishing images.
