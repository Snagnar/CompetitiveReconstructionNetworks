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
https://boxup.uni-potsdam.de/s/qRsJ9bYZCn4qCjX
```
using the following password: *xcnxAE7YFdseZ9M* and extract them e.g. to a directory called `datasets/`.

The `roadimages.zip` contains the dataset with images extracted from 3D lidar scans of streets in the town of essen.
The `panorama_images.zip` file contains the dataset of hand labled panorama images obtained from cameras mounted on the scanning vehicles.

