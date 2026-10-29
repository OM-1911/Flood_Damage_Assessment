# AI-Based Flood Damage Assessment System

A Django web application that uses a U-Net convolutional neural network to segment flooded areas from aerial/satellite images.

## Features

- Upload JPG/JPEG/PNG/WEBP flood images
- U-Net pixel-level flood segmentation
- Flood segmentation mask
- Red flood overlay
- Flood coverage percentage
- Low / Moderate / High / Severe classification
- Assessment history
- Django admin panel
- Training/test evaluation with Dice and IoU
- Training loss and Dice graphs

## Dataset

Put your extracted dataset in:

```text
dataset/
├── Image/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── Mask/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── metadata.csv
```

The provided dataset archive contains 290 matching image/mask pairs.

## Recommended Python

Python 3.10 or 3.11 is recommended for the TensorFlow/Django stack.

## Installation

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train the model

Before starting Django, place the dataset in `dataset/` and run:

```bash
python train_model.py
```

The script creates:

```text
model/flood_unet.keras
model/training_plots/loss.png
model/training_plots/dice.png
```

It also prints test Dice and IoU scores.

## Start the website

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## Optional admin

```bash
python manage.py createsuperuser
```

Then visit:

```text
http://127.0.0.1:8000/admin/
```

## Important

The severity value is a project-level estimate based on the percentage of pixels classified as flooded. It is not a certified engineering or emergency-management damage assessment.
