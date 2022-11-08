import argparse
import logging
import torch
import pathlib
import numpy as np
from classifier.config.defaults import cfg
from classifier.data.build import make_data_loaders
from classifier.logger import setup_logger
from classifier.trainer import Trainer
from classifier.models import build_model
from classifier.utils import to_cuda, load_best_checkpoint

np.random.seed(0)
torch.manual_seed(0)


def load_model(cfg):
    model = build_model(cfg)
    checkpoint_dir = pathlib.Path(cfg.OUTPUT_DIR)
    state_dict = load_best_checkpoint(checkpoint_dir)
    if state_dict:
        model.load_state_dict(state_dict=state_dict)
        print("loaded model")
    else: print("no model found")
    return model


def start_train(cfg):
    logger = logging.getLogger('classification.trainer')
    model = build_model(cfg)
    #model = load_model(cfg)
    model = to_cuda(model)
    #print(model)
    dataloaders = make_data_loaders(cfg)
    trainer = Trainer(
        cfg,
        model=model,
        dataloaders=dataloaders
        )
    trainer.train()
    return trainer.model


def get_parser():
    parser = argparse.ArgumentParser(description='Single Record MultiLine Detector Training With PyTorch')
    parser.add_argument(
        "config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger("Classifier", output_dir)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    model = start_train(cfg)

if __name__ == "__main__":
    main()
