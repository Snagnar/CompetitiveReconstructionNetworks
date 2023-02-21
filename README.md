# CompetitiveReconstructionNetworks
Implementation of the CRN paper

# install the requirements

It is recommended to use either `python3.8` or `python3.10`. It is also recommended to use a virtual environment for this project. If you are on a linux machine you can create one using the following commands:

```
sudo apt install python3-virtualenv && virtualenv venv
```

You can then activate this environment with 
```
source venv/bin/activate
```

You can then install the requirements for this repository using pip and the following command:
```
pip install -r requirements.txt
```

# Download of Datasets:

## MVTec AD

Download the zip file from the following url:

```
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
```

Extract the tarfile e.g. to a directory called `mvtec/`.

## Panorama and Road Images

Download the zip files from the following url:

```
https://boxup.uni-potsdam.de/s/4WZTcj3yi25iXYK
```
using the following password: *itJJag8WSxHAM9i* and extract them e.g. to a directory called `datasets/`.

The `roadimages.zip` contains the dataset with images extracted from 3D lidar scans of streets in the town of essen.
The `panorama_images.zip` file contains the dataset of hand labled panorama images obtained from cameras mounted on the scanning vehicles.

# Running experiments

To evaluate the CRN on MVTec, you can use the following command:
```bash
python train.py --mode train --training-steps 20000 --model crn --dataset MVTec --auto-set-name --dataset-path=datasets/mvtec/cable
```
You can substitute the dataset path to test out different categories. You can log metrics to `Weights & Biases` by adding the `--wandb` flag. However you need a W&B account and an API key for this.

To evaluate CRN on the annotated Panorama images, use the following command:
```bash
python train.py --mode train --dataset Panorama --dataset-path datasets/panorama --model crn
```


To evaluate CRN on the annotated road images, use the following command:
```bash
python train.py --mode train --dataset RoadImages --dataset-path datasets/roadimages --model crn
```

To view a full list of parameters, including e.g. the number of competitive units, optimizer or loss weights, run:
```
python train.py --help
```

# Inference

To run inference, you first need to train a model and store checkpoints using the `--checkpoint-path` flag, e.g. for road images:
```bash
python train.py --mode train --dataset RoadImages --dataset-path datasets/roadimages --model crn --checkpoint-path model_checkpoints
```

This stores the best CRN state in `model_checkpoints/`.

To calculate anomaly scores for one whole datasets as well as anomaly pictures, use the following command:

```bash
python train.py --mode train --dataset RoadImages --dataset-path datasets/roadimages --model crn --checkpoint-path model_checkpoints
```
