import torch
import collections
import pathlib
import typing
import time
import torch.nn as nn
from classifier.utils import to_cuda, save_checkpoint, load_best_checkpoint
from classifier.evaluate import calculate_mAP
from classifier.data.datasets import dereference_dict

class Trainer:
    """
    Trainer class for neural network
    """
    def __init__(
        self,
        cfg,
        model: nn.Module,
        dataloaders: typing.List[torch.utils.data.DataLoader],
        ):
        self.eval_step = getattr(cfg.TRAINER, "EVAL_STEP", 1)
        if getattr(cfg.TRAINER, "OPTIMIZER", "sgd") == "sgd":
            self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.TRAINER.LR,
            momentum=cfg.TRAINER.MOMENTUM,
            weight_decay = cfg.TRAINER.WEIGHT_DECAY
            )
        if getattr(cfg.TRAINER, "SCHEDULER", "multistep") == "multistep":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=cfg.TRAINER.LR_STEPS,
                gamma=cfg.TRAINER.GAMMA
            )
        if getattr(cfg.TRAINER, "ACTIVATION", "sigmoid") == "sigmoid":
            self.activation = nn.Sigmoid()
        self.batch_size = cfg.TRAINER.BATCH_SIZE
        self.epochs = cfg.TRAINER.EPOCHS
        self.log_path = cfg.OUTPUT_DIR + "/train.log"
        self.loss_criterion = nn.BCEWithLogitsLoss()
        self.model = to_cuda(model)
        self.train_data, self.val_data, self.test_data = dataloaders
        self.dereference_dict = dereference_dict(cfg.INPUT.NAME)
        #Metric tracking
        self.VALIDATION_LOSS = collections.OrderedDict()
        self.TEST_LOSS = collections.OrderedDict()
        self.TRAIN_LOSS = collections.OrderedDict()
        self.VALIDATION_RESULTS = collections.OrderedDict()
        self.TEST_RESULTS = collections.OrderedDict()

        #Checkpoint saving
        self.checkpoint_dir = pathlib.Path(cfg.OUTPUT_DIR)
        self.global_step = 0

    def save_model(self):
        """
        Saves current model in cfg.OUTPUT_DIR
        """
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            validation_losses = list(self.VALIDATION_LOSS.values())
            return validation_losses[-1] == min(validation_losses)
        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")
        save_checkpoint(state_dict, filepath, is_best_model())
    
    def load_best_model(self):
        state_dict = load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)
    
    def should_stop(self):
        """
        Early stop function (not yet implemented)
        """
        return False
    
    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        for epoch in range(self.epochs):
            self.epoch = epoch
            # Perform a full pass through all the training samples
            epoch_avg_loss = 0
            epoch_total_loss = 0
            self.model.train()
            i = 0
            print("learning rate: {}".format(self.optimizer.param_groups[0]['lr']))
            for x_batch, y_batch, _ in self.train_data:
                x_batch = to_cuda(x_batch)
                y_batch = to_cuda(y_batch)
                # X_batch is the time series after transforms.
                #Shape: [batch_size, TRANSFORM_OUTPUT_DIMS]
                # Y_batch is the audio label. Shape: [batch_size, num_classes]
                predictions = self.model.forward(x_batch)
                # Compute loss
                loss = self.loss_criterion(predictions, y_batch)
                self.TRAIN_LOSS[self.global_step] = loss.detach().cpu().item()
                epoch_total_loss += self.TRAIN_LOSS[self.global_step]
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient descent step
                self.optimizer.step()
                self.global_step += 1
                i += 1
                if self.should_stop():
                    print("Early stopping.")
                    return
            print("epoch average train loss: {}".format(epoch_total_loss/i))
            self.model.eval()
            if (epoch % self.eval_step) == 0:
                self.validation_epoch()
            self.model.train()
            self.save_model()
            self.lr_scheduler.step()
        self.end_evaluation()
    
    def compute_mAP(self, dataloader, plot_curves=False):
        """
        Computes the mean Average Precision for a given dataloader
        for a multilabel classifier.
        """
        average_loss = 0
        total_loss = 0
        total_batches = 0
        all_predictions = torch.Tensor()
        all_targets = torch.Tensor()
        with torch.no_grad():
            for x_batch, y_batch, _ in dataloader:
                x_batch = to_cuda(x_batch)
                y_batch = to_cuda(y_batch)
                outputs = self.model(x_batch)
                total_batches += 1
                total_loss += self.loss_criterion(outputs,y_batch)
                output_probs = self.activation(outputs)
                output_probs = output_probs.cpu()
                y_batch = y_batch.cpu()
                all_predictions = torch.cat(
                    (all_predictions, output_probs),
                    dim = 0
                )
                all_targets = torch.cat(
                    (all_targets, y_batch),
                    dim = 0
                )
            average_precisions = calculate_mAP(
                all_predictions,
                all_targets,
                plot_curves,
                self.dereference_dict
                )
            average_loss = total_loss/total_batches
        return average_precisions, average_loss

    def validation_epoch(self, plot_pr_curves=False):
        """
            Computes the loss/accuracy for validation dataset
            Can conditionally plot precision-recall curves
            for each class if plot_pr_curves is set to true
        """
        average_precisions, average_loss = self.compute_mAP(
            self.val_data,
            plot_curves=plot_pr_curves
        )
        self.VALIDATION_RESULTS[self.global_step] = average_precisions
        self.VALIDATION_LOSS[self.global_step] = average_loss
        print(
            f"Epoch: {self.epoch + 1}\n",
            f"Global step: {self.global_step}\n",
            f"Validation Loss: {average_loss}\n",
            sep="\t"
        )
        self.print_precisions(average_precisions)
        log = open(self.log_path,"a+")
        log.write(f"time: {str(time.time())}\n")
        log.write(f"Epoch: {self.epoch}\n")
        log.write(f"Global step: {self.global_step}\n")
        log.write(f"Validation average precisions: {average_precisions}\n")
        log.write(f"Validation Loss: {average_loss}\n")
        log.close()

    def end_evaluation(self):
        """
        Evaluation performed after training
        Will load the best model, currently set to the model
        with the lowest validation loss.
        """
        self.load_best_model()
        self.model.eval()
        self.validation_epoch()
        average_precisions, average_loss = self.compute_mAP(self.test_data)
        print(
            f"Epoch: {self.epoch + 1}\n",
            f"Global step: {self.global_step}\n",
            f"Test Loss: {average_loss}\n",
            sep="\t")
        self.print_precisions(average_precisions)
        log = open(self.log_path,"a+")
        log.write(f"time: {str(time.time())}\n")
        log.write(f"Epoch: {self.epoch}\n")
        log.write(f"Global step: {self.global_step}\n")
        log.write(f"Test loss: {average_loss}\n")
        log.close()
    
    def print_precisions(self, precisions):
        print("Average Precisions\n")
        for key in precisions:
            print(f"{key} : {precisions[key]}\n")
