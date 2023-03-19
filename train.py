import os
import numpy as np
import sys
import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from torch.utils.data import DataLoader
import logging
from training.utils import check_paths, set_logging
from training.dataset import AnnotatedRoadImageDataset, MVTecDataset, PanoramaDataset
from training.double_skip import DAGAN
from training.crn import CompetitiveReconstructionNetwork
from pytorch_lightning.loggers import WandbLogger
import wandb



def add_argparse_args(parser):
    model_parser = parser.add_argument_group("Model Arguments")
    model_parser.add_argument("--contextual-loss-weight", type=float, default=40.0)
    model_parser.add_argument("--adversarial-loss-weight", type=float, default=1.0)
    model_parser.add_argument("--discriminator-reconstruction-loss-weight", type=float, default=1.0)

    model_parser.add_argument("--reconstruction-weight", type=float, default=0.5)
    model_parser.add_argument("--max-network-depth", type=int, default=8)
    model_parser.add_argument("--generator-network-depth", type=int, default=4)
    model_parser.add_argument("--imsize", type=int, default=128)
    model_parser.add_argument("--feedback-weight", type=float, default=1.0)
    model_parser.add_argument("--discrimination-weight", type=float, default=2.0)
    model_parser.add_argument("--improved", type=bool, default=True,
                                help="if activated, the crns use the improved unet version.")
    model_parser.add_argument("--use-dropout", type=bool, default=False,
                                help="if activated, competitive units are trained on differing data samples.")
    model_parser.add_argument("--image-output-interval", type=float, default=10)
    model_parser.add_argument("--lr-scheduler", type=str,
                        help="name of the lr scheduler to use. default to none")
    model_parser.add_argument("--norm", type=str, default="batch",
                        help="name of the lr scheduler to use. default to none")
    model_parser.add_argument("--lr", type=float, default=0.0001,
                        help="initial learning rate (default: 0.0001)")
    model_parser.add_argument('--lr-factor', default=0.1, type=float,
                        help='learning rate decay ratio. this is used only by the step and exponential lr scheduler')
    model_parser.add_argument('--lr-steps', nargs="*", default=[30, 60, 90],
                        help='list of learning rate decay epochs as list. this is used only by the step scheduler')
    model_parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9. only used for sgd optimizer')
    model_parser.add_argument('--optimizer', type=str, default="adam",
                        help='the optimizer to use. default is adam.')
    model_parser.add_argument("--image-output-path", type=str, default=None)
    model_parser.add_argument("--anomaly-score-file", type=str, default=None)


def main(args):
    if args.seed is not None:
        seed_everything(args.seed, workers=True)
    torch.autograd.set_detect_anomaly(True)
    
    if args.demo:
        logging.warning("DEMO MODE ACTIVATED! training with batch size 2 for 100 training steps and image size 32x32. Networks have a maximum depth of 3 and only 3 competitive units are used.")
        args.batch_size = 1
        args.training_steps = 50
        args.max_network_depth = 2
        args.num_competitive_units = 3
        args.imsize = 32
    
    set_logging(args.log_file, args.log_level, args.log_stdout)

    data_path = Path(args.dataset_path)
    check_paths(data_path, names=["image directory"])
    if args.dataset == "RoadImages":
        train_dataset = AnnotatedRoadImageDataset(data_path, train=True, imsize=args.imsize)
        test_dataset = AnnotatedRoadImageDataset(data_path, train=False, imsize=args.imsize)
    elif args.dataset == "MVTec":
        train_dataset = MVTecDataset(data_path, train=True, imsize=args.imsize)
        test_dataset = MVTecDataset(data_path, train=False, imsize=args.imsize)
    elif args.dataset == "Panorama":
        train_dataset = PanoramaDataset(data_path, train=True)
        test_dataset = PanoramaDataset(data_path, train=False)
    
    if args.demo:
        train_indices = np.random.permutation(len(train_dataset))[:50]
        test_indices = np.random.permutation(len(test_dataset))[:50]
        train_dataset = [train_dataset[i] for i in train_indices]
        test_dataset = [test_dataset[i] for i in test_indices]

    logging.info("Dataset created!")
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, persistent_workers=args.num_workers)
    val_loader = DataLoader(test_dataset, max(args.batch_size, 128 * int(not args.demo)), num_workers=args.num_workers, pin_memory=True, shuffle=False, persistent_workers=args.num_workers)

    name = None
    logger = True
    if args.wandb:
        logging.info("using weights & biases as logger")
        name = args.experiment
        logger = WandbLogger(project="road-quality-evaluation", log_model=False, name=name)
        wandb.init(config=args, name=name)
        wandb.config.update({"run_command": " ".join(sys.argv)})

    epochs = args.epochs if args.epochs is not None else int(args.training_steps / (len(train_dataset) / args.batch_size)) + 1
    args.epochs = epochs
    logging.info("creating trainer....")
    trainer = Trainer(
        accelerator="gpu" if (not args.cpu and torch.cuda.device_count() > 0) else "cpu",
        devices=max(torch.cuda.device_count(), 2) if not args.cpu else os.cpu_count() // 4,
        max_epochs=epochs,
        auto_select_gpus=True,
        logger=logger,
        enable_checkpointing=args.checkpoint_path is not None,
        callbacks = [
            ModelCheckpoint(dirpath=args.checkpoint_path, save_last=False, mode="max", monitor="metrics/max_roc_auc", save_top_k=1),
        ] if args.checkpoint_path is not None else [],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        max_steps=args.training_steps,
        deterministic=args.seed is not None,
        num_sanity_val_steps=0,
        precision=16,
        benchmark=args.seed is None,
    )
    logging.info("Trainer created!")
    
    input_shape = train_dataset[0].unsqueeze(0).shape
    model_class = CompetitiveReconstructionNetwork if args.model == "crn" else DAGAN
    if args.mode == "train":
        logging.info("training model....")
        if args.model_input is None:
            logging.info("Training new model...")
            model = model_class(input_shape, **vars(args))
        else:
            logging.info("continuing training....")
            model = model_class.load_from_checkpoint(args.model_input)
        
        logging.info("Model created!")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=[val_loader])
    
    elif args.mode == "inference":

        if args.dataset == "RoadImages":
            train_dataset = AnnotatedRoadImageDataset(data_path, train=True, imsize=args.imsize, inference=True)
            test_dataset = AnnotatedRoadImageDataset(data_path, train=False, imsize=args.imsize, inference=True)
        elif args.dataset == "MVTec":
            train_dataset = MVTecDataset(data_path, train=True, imsize=args.imsize, inference=True)
            test_dataset = MVTecDataset(data_path, train=False, imsize=args.imsize, inference=True)
        elif args.dataset == "Panorama":
            train_dataset = PanoramaDataset(data_path, train=True, inference=True)
            test_dataset = PanoramaDataset(data_path, train=False, inference=True)
        data_path = Path(args.dataset_path)
        if args.image_output_path is None:
            image_output_path = Path("inference")
            logging.info(f"using default image output path: {image_output_path.resolve()}")
        else:
            image_output_path = Path(args.image_output_path)
            logging.info(f"using specified image output path: {image_output_path.resolve()}")
        check_paths(data_path, names=["image directory"])
        if args.anomaly_score_file is None:
            results = image_output_path / "anomaly_scores.csv"
            logging.info(f"using default anomaly score file: {results.resolve()}")
        else:
            results = Path(args.anomaly_score_file)
            logging.info(f"using specified anomaly score file: {results.resolve()}")
        results.parent.mkdir(parents=True, exist_ok=True)

        # load model from specified checkpoint
        model = model_class.load_from_checkpoint(args.model_input)
        model.image_output_path = image_output_path
        model.image_output_path.mkdir(parents=True, exist_ok=True)

        device = 'cuda' if torch.cuda.device_count() > 0 and not args.cpu else 'cpu'
        logging.info(f"using device: {device}")
        model = model.to(device)

        anomaly_scores = model.inference(test_dataset, device, shall_clean=True)
        anomaly_scores += model.inference(train_dataset, device)
        with results.open("w") as result_file:
            result_file.write("file_name,anomaly_score\n")
            for file_name, score in anomaly_scores:
                result_file.write(f"{file_name},{score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training of Network for Road anomaly detection")
    

    parser.add_argument("--log-level", type=str, default="info",
                     choices=["debug", "info", "warning", "error", "critical"],
                     help="log level for logging message output")
    parser.add_argument("--log-file", type=str, default="log.log",
                     help="output file path for logging. default to stdout")
    parser.add_argument("--log-stdout", action="store_true", default=True,
                     help="toggles force logging to stdout. if a log file is specified, logging will be "
                     "printed to both the log file and stdout")
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--model", choices=["dagan", "crn"], required=False, default="crn")
    parser.add_argument("--pipeline", type=str, required=False, help="path to pipeline json")
    parser.add_argument("--experiment", type=str, required=False, help="name of experiment (used for wandb)")
    parser.add_argument("--training-steps", type=int, required=False, default=20000, help="number of steps to train")
    parser.add_argument("--epochs", type=int, required=False, help="number of epochs")
    parser.add_argument("--batch-size", type=int, required=False, default=64)
    parser.add_argument("--num-competitive-units", type=int, default=12, required=False, help="number of competitive units used for crn")
    parser.add_argument("--num-workers", type=int, required=False, default=0, help="number of workers for dataloader")
    parser.add_argument("--dataset-path", type=str, required=False, help="path to dataset")
    parser.add_argument("--model-input", type=str, required=False, help="path to model (for inference)")
    parser.add_argument("--dataset", type=str, choices=["RoadImages", "MVTec", "Panorama"], required=False, help="name of dataset")
    parser.add_argument("--checkpoint-path", type=str, required=False, default=None, help="path to store the model checkpoints")
    parser.add_argument("--cpu", action="store_true", help="force training on cpu")
    parser.add_argument("--auto-set-name", action="store_true", help="sets automatically the name of the experiment")
    parser.add_argument("--wandb", action="store_true", help="activate weights and biases logging")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("--wandb-hyperparameter-sweep", action="store_true", help="activate weights and biases hyperparameter sweep")
    parser.add_argument("--demo", action="store_true", help="activates demo mode, which will train the networks with a fraction of the data")
    Trainer.add_argparse_args(parser)
    add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
