# CompetitiveReconstructionNetworks

Implementation of the anomaly detection technique described in "Detecting Road Damages in Mobile Mapping Point Clouds using Competitive Reconstruction Networks".

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

For this repository it is highly recommended to use a graphics card with at least 12 GB of vram. If your graphics card does not have enough memory, you might have to reduce the number of competitive units in the commands used below using the `--num-competitive-units` parameter. If you are using a graphics card, you also need to install `cuda>10.0` together with pytorch, e.g. `pip install torch==1.12.1+cu116` if you have `cuda` 11.6 installed. You can also train using a CPU but be aware that this is VERY slow (>10 minutes per epoch).

# Download of Datasets:

## MVTec AD

Download the zip file from the following url:

```
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
```
The dataset has an extracted size of roughly 5 gb.
Extract the tarfile e.g. to a directory called `datasets/mvtec/`.

Alternatively, if the link is broken, use the following link to apply for a new download link:
```
https://www.mvtec.com/company/research/datasets/mvtec-ad/
```

## Panorama and Road Images

Download the zip files from the following url:

```
https://doi.org/10.5281/zenodo.7681876
```
and extract them e.g. to a directory called `datasets/`. The two datasets have a combined size of roughly 280 mb.

The `roadimages.zip` contains the dataset with images extracted from 3D lidar scans of streets in the town of essen.
The `panorama_images.zip` file contains the dataset of hand labled panorama images obtained from cameras mounted on the scanning vehicles.

# Running experiments

To evaluate the CRN on MVTec, you can use the following command:
```bash
python train.py --mode train --training-steps 20000 --model crn --dataset MVTec --auto-set-name --dataset-path=datasets/mvtec/cable --seed 41020
```
You can substitute the dataset path to test out different categories. You can log metrics to `Weights & Biases` by adding the `--wandb` flag. However you need a W&B account and an API key for this.

To evaluate CRN on the annotated Panorama images, use the following command:
```bash
python train.py --mode train --dataset Panorama --dataset-path datasets/panorama --model crn --seed 41020
```


To evaluate CRN on the annotated road images, use the following command:
```bash
python train.py --mode train --dataset RoadImages --dataset-path datasets/roadimages --model crn --seed 41020
```

To view a full list of parameters, including e.g. the number of competitive units, optimizer or loss weights, run:
```
python train.py --help
```

## Demo mode:

As the hardware requirements to run the experiments are rather high (the results of the papers were obtained using a A100 graphics card), you can enable demo mode with the flag `--demo`. This sets the batch size to 2, the maximum network depth to 3, the image size used for training to 32x32, the number of competitive units to 3 and the maximum number of training steps to 50. 

Using only a an intel i7 cpu, the following command took roughly 20 min. The obtained maximum roc auc was 0.78.
```
python train.py --mode train --dataset RoadImages --dataset-path datasets/roadimages --model crn --demo --num-workers 4 --cpu --seed 41020
```

# Inference

To run inference, you first need to train a model and store checkpoints using the `--checkpoint-path` flag, e.g. for road images:
```bash
python train.py --mode train --dataset RoadImages --dataset-path datasets/roadimages --model crn --checkpoint-path model_checkpoints
```
This stores the best CRN state in `model_checkpoints/`.

To calculate anomaly scores for one whole datasets as well as anomaly pictures, use the following command:
```bash
python train.py --mode inference --dataset Panorama --dataset-path datasets/panorama_converted --model crn --model-input "model_checkpoints/last.ckpt" --image-output-path inference_images/
```
For `--model-input` make sure to give the correct path to the last model checkpoint.

To make inference for different datasets, (RoadImages, MVTec or Panorama) adjust the `--dataset` and `--dataset-path` parameters accordingly.

After inference, you can look at the pictures generated in the `diff` directory to make a qualitative evaluation of the results.

# Hyperparameter Optimization

The `wandb` sweep configuration used in the paper to optain an optimized set of hyperparameters can be found in the `hyperparameter_optimization_sweep.yml`.
To start a hyperparameter optimization run, use the following command:
```
wandb sweep hyperparameter_optimization_sweep.yml
```
Note: you need a W&B account for this.
This gives you an `wandb agent ...` command that you can execute on a machine equipped with suitable hardware (i.e. a graphics card with 12GB+ vram). The
agent will then test different hyperparameter configurations using the bayes optimization algorithm. You can start multiple agents on different machines.

You can also use the `wandb_sweep.yml` file to simply test all categories of the MVTec dataset in a row. For this, simply swap the sweep configuration
filename above with `wandb_sweep.yml`.
