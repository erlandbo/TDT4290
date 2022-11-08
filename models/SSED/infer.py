"""
Scanline Sound Event Detection inference program
Written by Bendik Bogfjellmo in 2021
(github.com/bendikbo) (bendik.bogfjellmo@gmail.com)
Feel free to reuse, but let a brother get in them mentions

The main idea of the inference stuff is to do n hops per
classifying window. Meaning that each "subwindow" has n
classifications made on it, and then just average all these
confidence scores and just threshold those to get get them bools
"""
import torch
import pandas as pd
import pathlib
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import sigmoid
#Used to dereference from active neuron to class name
from classifier.data.datasets import dereference_dict
from classifier.models import build_model
from classifier.config.defaults import cfg
from classifier.utils import to_cuda
from classifier.utils import load_best_checkpoint
from classifier.data.datasets.window_slide import WindowSlide
import time


class Inferencer:
    def __init__(
        self,
        cfg,
        audio_files: list
        ):
        self.start_time = time.time()
        self.cfg = cfg
        self.audio_files = audio_files
        self.dataloaders = []
        self.dereference_dict = dereference_dict(self.cfg.INPUT.NAME)
        for audio_file in audio_files:
            self.add_dataloader(audio_file)
        self.model = self.load_model()
        self.model.eval()
        to_cuda(self.model)
        self.output_dir = self.cfg.INFERENCE.OUTPUT_DIR

    def add_dataloader(self, audio_file):

        dataset = WindowSlide(self.cfg, audio_file)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.INFERENCE.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.INFERENCE.NUM_WORKERS
        )
        self.dataloaders.append(dataloader)
    
    def load_model(self):

        model = build_model(self.cfg)
        checkpoint_dir = pathlib.Path(self.cfg.OUTPUT_DIR)
        state_dict = load_best_checkpoint(checkpoint_dir)
        model.load_state_dict(state_dict=state_dict)
        return model
    
    def infer(self):
        for audio_file, dataloader in zip(self.audio_files, self.dataloaders):
            self.start_time = time.time() #For performance measurements
            #output filename
            output_filename = audio_file.split("/")[-1].split(".")[0]
            #Instantiating predictions for entire audio file
            all_raw_preds = torch.zeros(
                len(dataloader.dataset),
                len(self.dereference_dict)
            )
            #Size [Predictions_per_file, Num_classes]
            all_raw_preds = to_cuda(all_raw_preds)
            with torch.no_grad():
                for x_batch, indices in tqdm(dataloader):
                    x_batch = to_cuda(x_batch)
                    #Have to convert tensor to long for indexing
                    indices = indices.long()
                    indices = to_cuda(indices)
                    preds = self.model(x_batch)
                    preds = sigmoid(preds)
                    #Just to make absolutely sure everything's in order
                    all_raw_preds[indices] = preds
            num_hops = self.cfg.INFERENCE.HOPS_PER_WINDOW
            #processed_preds.size() == (predictions_per_file + num_hops - 1, num_classes)
            processed_preds = torch.zeros(
                (all_raw_preds.size()[0]+num_hops - 1,
                all_raw_preds.size()[1])
            )
            #For loop to make processes preds into moving average
            #of raw preds based on number of hops per class window
            all_raw_preds = all_raw_preds.cpu()
            processed_preds = processed_preds.cpu()
            for hop_no in range(num_hops):
                #Have to add if statement for last hop, since
                #apparantly, there's no elegant way to do this
                if hop_no < (num_hops - 1):
                    processed_preds[hop_no:-num_hops + hop_no + 1,:] \
                    += all_raw_preds[:,:]/num_hops
                else:
                    processed_preds[hop_no:,:]\
                    += all_raw_preds[:,:]/num_hops
            if num_hops > 1:
                num_labels=processed_preds.size()[-1]
                #This is done to fix edge cases in moving mean method
                numerator=torch.linspace(1,num_hops-1, num_hops-1)
                denominator = torch.linspace(num_hops-1, 1, num_hops-1)
                edge_multiplier=torch.div(numerator, denominator)
                #Have to repeat multiplier for each label type
                edge_multiplier=\
                edge_multiplier.view(-1,1).repeat(1,num_labels).view(num_hops-1,num_labels)
                processed_preds[-num_hops + 1:,:]+=\
                torch.mul(
                    edge_multiplier,
                    processed_preds[-num_hops+1:,:]
                    )
                #Flip it around and bring it back
                edge_multiplier = torch.div(denominator, numerator)
                edge_multiplier=\
                edge_multiplier.view(-1,1).repeat(1,num_labels).view(num_hops-1,num_labels)
                processed_preds[0:num_hops-1,:]+=\
                torch.mul(
                    edge_multiplier,
                    processed_preds[:num_hops-1,:]
                )
            self.format_and_write(output_filename, processed_preds)
    
    def format_and_write(
        self,
        output_filename : str,
        processed_preds : torch.Tensor
        ):

        predictions = []
        threshold = self.cfg.INFERENCE.THRESHOLD
        for label in range(processed_preds.size()[1]):
            label_name = self.dereference_dict[label]
            label_preds = processed_preds[:,label]
            pos_preds = torch.zeros_like(label_preds)
            pos_preds[label_preds >= threshold] = 1
            i = 0
            pred_start = 0.0
            pred_stop = 0.0
            active_pred = False
            hop_length_in_seconds = \
                (self.cfg.INPUT.RECORD_LENGTH//self.cfg.INFERENCE.HOPS_PER_WINDOW)/ \
                self.cfg.INPUT.SAMPLE_FREQ
            num_preds = int(pos_preds.size()[0])
            #Couldn't figure out a prettier way to do this
            while i < num_preds:
                if pos_preds[i]:
                    if not active_pred:
                        pred_start = i*hop_length_in_seconds
                    active_pred = True
                elif pos_preds[i-1] and active_pred:
                    active_pred = False
                    pred_stop = i*hop_length_in_seconds
                    predictions.append((label_name, pred_start, pred_stop))
                i+=1
        if self.cfg.INFERENCE.OUTPUT_FORMAT == "csv":
            self.write_to_csv(predictions, output_filename)
        elif self.cfg.INFERENCE.OUTPUT_FORMAT == "audacity":
            self.write_audacity_labels(predictions, output_filename)
    
    def write_to_csv(
        self,
        predictions,
        output_filename
        ):
        df = pd.DataFrame(
            predictions,
            columns=["class", "onset", "offset"]
            )
        pathlib.Path(cfg.INFERENCE.OUTPUT_DIR).mkdir(
            parents=True,
            exist_ok=True
        )
        output_path = self.cfg.INFERENCE.OUTPUT_DIR + output_filename + ".csv"
        df.to_csv(output_path)
        print(f"\n\nFinished inference on: {output_filename}")
        print(f"time used: {time.time() - self.start_time}")
        print(f"ouput stored as: {output_path}")

    def write_audacity_labels(
        self,
        predictions,
        output_filename
        ):
        pathlib.Path(cfg.INFERENCE.OUTPUT_DIR).mkdir(
            parents=True,
            exist_ok=True
        )
        stuff_to_write = ""
        for prediction in predictions:
            first_line = f"{prediction[1]}	{prediction[2]}	{prediction[0]}\n"
            second_line = f"\\	000.000	000.000\n"
            stuff_to_write += first_line + second_line
        #Audacity label files are .txt extension
        output_path = self.cfg.INFERENCE.OUTPUT_DIR + output_filename + ".txt"
        out_file = open(output_path, 'w+')
        out_file.write(stuff_to_write)
        out_file.close()
        print(f"\n\nFinished inference on: {output_filename}")
        print(f"time used: {time.time() - self.start_time}")
        print(f"ouput stored as: {output_path}")



def get_parser():
    parser = argparse.ArgumentParser(
        description='Sound Event Detection inference')
    parser.add_argument(
        "config_file",
        default="",
        metavar="config_file",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "audio_files",
        help="Audio files to run inference on",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    audio_files = args.audio_files
    cfg.freeze()
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))
    inferencer = Inferencer(cfg, audio_files)
    inferencer.infer()


if __name__ == "__main__":
    main()
