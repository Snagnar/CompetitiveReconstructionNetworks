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
from training.double_skip import RoadAnomalyDetector
from training.crn import CompetitiveReconstructionNetwork
from pytorch_lightning.loggers import WandbLogger
import wandb



def add_argparse_args(parser):
    model_parser = parser.add_argument_group("Model Arguments")
    model_parser.add_argument("--contextual-loss-weight", type=float, default=40.0)
    model_parser.add_argument("--adversarial-loss-weight", type=float, default=1.0)
    model_parser.add_argument("--discriminator-reconstruction-loss-weight", type=float, default=1.0)

    model_parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    model_parser.add_argument("--network-depth", type=int, default=4)
    model_parser.add_argument("--generator-network-depth", type=int, default=4)
    model_parser.add_argument("--warmup-steps", type=int, default=0)
    model_parser.add_argument("--stride", type=int, default=4)
    model_parser.add_argument("--feedback-group-size", type=int, default=0)
    model_parser.add_argument("--abnormal-class-idx", type=int, default=0)
    model_parser.add_argument("--abnormal-class-name", type=str, default=0)
    model_parser.add_argument("--arch", type=int, default=0)
    model_parser.add_argument("--training-mode", type=int, default=0)
    model_parser.add_argument("--minimum-learning-units", type=int, default=4)
    model_parser.add_argument("--imsize", type=int, default=128)
    model_parser.add_argument("--gamma_crn", type=float, default=0.75)
    model_parser.add_argument("--gamma_reconstruction", type=float, default=0.75)
    model_parser.add_argument("--gamma_feedback", type=float, default=0.75)
    model_parser.add_argument("--gamma_discrimination", type=float, default=0.75)
    model_parser.add_argument("--feedback-weight", type=float, default=1.0)
    model_parser.add_argument("--discrimination-weight", type=float, default=1.0)
    model_parser.add_argument("--self-discrimination", type=bool, default=False,
                                help="if activated the feedback loss for one unit will contain the discrimination of the own reconstruction.")
    model_parser.add_argument("--dynamic-loss-weights", type=bool, default=False,
                                help="if activated loss weights are determined dynamically.")
    model_parser.add_argument("--improved", type=bool, default=False,
                                help="if activated loss weights are determined dynamically.")
    model_parser.add_argument("--competitive-cross-validation", type=bool, default=False,
                                help="if activated, competitive units are trained on differing data samples.")
    model_parser.add_argument("--ignore-feedback", type=bool, default=False,
                                help="if activated, competitive units are trained on differing data samples.")
    model_parser.add_argument("--bagging", type=bool, default=False,
                                help="if activated, competitive units are trained on differing data samples.")
    model_parser.add_argument("--use-dropout", type=bool, default=False,
                                help="if activated, competitive units are trained on differing data samples.")
    model_parser.add_argument("--feedback-loss-reduction", choices=["mean", "sum"], default="sum", help="reduction of feedback losses")
    model_parser.add_argument("--discrimination-loss-reduction", choices=["mean", "sum"], default="sum", help="reduction of discrimination losses")
    model_parser.add_argument("--gamma_disc", type=float, default=0.75)
    model_parser.add_argument("--gamma_gen", type=float, default=0.75)
    model_parser.add_argument("--image-output-interval", type=float, default=10)
    model_parser.add_argument("--k-lambda", type=float, default=0.001)
    model_parser.add_argument("--lr-scheduler", type=str,
                        help="name of the lr scheduler to use. default to none")
    model_parser.add_argument("--norm", type=str, default="instance",
                        help="name of the lr scheduler to use. default to none")
    model_parser.add_argument("--lr", type=float, default=0.1,
                        help="initial learning rate (default: 0.1)")
    model_parser.add_argument('--lr-factor', default=0.1, type=float,
                        help='learning rate decay ratio. this is used only by the step and exponential lr scheduler')
    model_parser.add_argument('--lr-steps', nargs="*", default=[30, 60, 90],
                        help='list of learning rate decay epochs as list. this is used only by the step scheduler')
    model_parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9. only used for sgd optimizer')
    model_parser.add_argument('--optimizer', type=str, default="adam",
                        help='the optimizer to use. default is adam.')
    model_parser.add_argument("--image-output-path", type=str, default=None)


def main(args):
    seed_everything(args.seed, workers=True)
    torch.autograd.set_detect_anomaly(True)
    
    
    set_logging(args.log_file, args.log_level, args.log_stdout)

    logging.info("Training model...")
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

    logging.info("Dataset created!")
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(test_dataset, max(args.batch_size, 128), num_workers=args.num_workers, pin_memory=True, shuffle=False)

    name = None
    logger = True
    if args.wandb:
        logging.info("using weights & biases as logger")
        name = args.experiment
        logger = WandbLogger(project="road-quality-evaluation", log_model=False, name=name)
        wandb.init(config=args, name=name)
        wandb.config.update({"run_command": " ".join(sys.argv)})

    args.training_steps += args.warmup_steps
    epochs = args.epochs if args.epochs is not None else int(args.training_steps / (len(train_dataset) / args.batch_size)) + 1
    args.epochs = epochs
    logging.info("creating trainer....")
    trainer = Trainer(
        accelerator="gpu" if not args.cpu else "cpu",
        gpus=torch.cuda.device_count() if not args.cpu else None,
        max_epochs=epochs,
        auto_select_gpus=True,
        logger=logger,
        enable_checkpointing=args.checkpoint_path is not None,
        callbacks = [
            ModelCheckpoint(dirpath=args.checkpoint_path, save_last=False, mode="max", monitor="metrics/max_roc_auc", save_top_k=1),
        ],
        log_every_n_steps=2,
        check_val_every_n_epoch=1,
        max_steps=args.training_steps,
        deterministic=True,
        num_sanity_val_steps=0,
        precision=16,
        benchmark=False,
    )
    logging.info("Trainer created!")
    
    example_input = iter(train_loader).next()
    input_shape = example_input.shape
    model_class = CompetitiveReconstructionNetwork if args.model == "crn" else RoadAnomalyDetector
    if args.mode == "train":
        if args.model_input is None:
            logging.info("Training new model...")
            if args.model == "original":
                args.arch = 0
            elif args.model == "double-skip":
                args.arch = 1
            elif args.model == "balanced-generator":
                args.arch = 2
            elif args.model == "crn":
                args.arch = 3
            if args.training_function == "original":
                args.training_mode = 0
            elif args.training_function == "double-skip":
                args.training_mode = 1
            elif args.training_function == "balanced-generator":
                args.training_mode = 2
            elif args.training_function == "crn":
                args.training_mode = 3
            if args.validation_function == "original":
                args.validation_function = 0
            elif args.validation_function == "double-skip":
                args.validation_function = 1
            elif args.validation_function == "balanced-generator":
                args.validation_function = 2
            elif args.validation_function == "only-disc":
                args.validation_function = 4
            model = model_class(input_shape, **vars(args))
        else:
            logging.info("continuing training....")
            model = model_class.load_from_checkpoint(args.model_input)
        
        logging.info("Model created!")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=[val_loader])
    
    elif args.mode == "inference":
        data_path = Path(args.dataset_path)
        check_paths(data_path, names=["image directory"])
        results = Path(args.anomaly_score_file)
        results.parent.mkdir(parents=True, exist_ok=True)

        # load model from specified checkpoint
        model = model_class.load_from_checkpoint(args.model_input)
        model.image_output_path = Path(args.image_output_path)
        model.image_output_path.mkdir(parents=True, exist_ok=True)

        device = 'cuda' if torch.cuda.device_count() > 0 and not args.cpu else 'cpu'
        logging.info(f"using device: {device}")
        model = model.to(device)

        anomaly_scores = model.inference(val_loader, device, shall_clean=True)
        anomaly_scores += model.inference(train_loader, device)
        with results.open("w") as result_file:
            result_file.write("file_name,anomaly_score\n")
            for image_nr, score in zip(range(len(test_dataset)), anomaly_scores):
                result_file.write(f"val-{image_nr},{score}\n")
            for image_nr, score in zip(range(len(train_dataset)), anomaly_scores[len(test_dataset):]):
                result_file.write(f"train-{image_nr},{score}\n")

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
    parser.add_argument("--model", choices=["double-skip", "crn", "original", "balanced-generator"], required=False, default="double-skip")
    parser.add_argument("--training-function", choices=["double-skip", "crn", "original", "balanced-generator"], required=False, default="double-skip")
    parser.add_argument("--validation-function", choices=["double-skip", "crn", "original", "balanced-generator", "only-disc"], required=False, default="double-skip")
    parser.add_argument("--pipeline", type=str, required=False, help="path to pipeline json")
    parser.add_argument("--experiment", type=str, required=False, help="path to pipeline json")
    parser.add_argument("--training-steps", type=int, required=False, default=20000, help="path to pipeline json")
    parser.add_argument("--epochs", type=int, required=False, help="path to pirpeline json")
    parser.add_argument("--batch-size", type=int, required=False, default=2, help="path to pipeline json")
    parser.add_argument("--num-competitive-units", type=int, default=3, required=False, help="path to pipeline json")
    parser.add_argument("--num-workers", type=int, required=False, default=8, help="path to pipeline json")
    parser.add_argument("--dataset-path", type=str, required=False, help="path to pipeline json")
    parser.add_argument("--model-input", type=str, required=False, help="path to pipeline json")
    parser.add_argument("--dataset", type=str, required=False, help="path to pipeline json")
    parser.add_argument("--checkpoint-path", type=str, required=False, default=None, help="path to pipeline json")
    parser.add_argument("--conv-bias", type=bool, required=False, default=False, help="activate weights and biases logging")
    parser.add_argument("--cpu", action="store_true", help="activate weights and biases logging")
    parser.add_argument("--auto-set-name", action="store_true", help="activate weights and biases logging")
    parser.add_argument("--wandb", action="store_true", help="activate weights and biases logging")
    parser.add_argument("--seed", type=int, default=41020, help="seed")
    parser.add_argument("--wandb-hyperparameter-sweep", action="store_true", help="activate weights and biases hyperparameter sweep")
    Trainer.add_argparse_args(parser)
    add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
