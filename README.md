# HarvestGuard
Personal Streamlit inference interface for a plant disease classifier developed as a four-member team project.

Original team project: https://github.com/iamaniket1/harvestguard-crop-disease-detection

## Model
Fine-tuned MobileNetV2 CNN, PlantVillage color dataset, 38 healthy and disease classes. RGB input: 224 × 224, pixels 0–255; preprocessing is embedded in the saved model.

## Reported test results
| Metric | Value |
|---|---:|
| Accuracy | 96.67% |
| Top-3 accuracy | 99.63% |
| Macro F1 | 96.01% |

Source: shared model metadata. Results have not been independently reproduced.

## Setup
Use Python 3.10 or 3.11.
```bash
pip install -r requirements.txt
```
Download the [shared trained model](https://github.com/iamaniket1/harvestguard-crop-disease-detection/blob/main/backend/models/harvestguard.keras) and place it at `models/harvestguard.keras`. The metadata is included.

```bash
streamlit run app.py
# Or use the CLI:
python predict.py leaf.jpg
```

## Limitations
Supports only the 38 trained classes. Field-photo performance is unverified. Unsupported images may still receive a trained label. Model scores are not a diagnosis. Original training code and detailed evaluation artifacts are not included here. The personal inference interface has not yet been runtime-tested.
