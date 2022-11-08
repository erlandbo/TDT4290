import torch
import torch.nn as nn


class TargetTransform(nn.Module):
    """
    Target transformation for a "scanline" classifier,
    strategy is that any "scanline" containing more than a threshold
    part of it as a value for a class should be regarded as positive.
    """
    def __init__(self, cfg):
        super(TargetTransform, self).__init__()
        if hasattr(cfg.INPUT.TRANSFORM, "SPECTROGRAM"):
            self.length = cfg.INPUT.TRANSFORM.SPECTROGRAM.RESOLUTION[0]
        else:
            self.length = cfg.INPUT.RECORD_LENGTH
        self.threshold = cfg.MODEL.THRESHOLD
        self.num_classes = cfg.MODEL.NUM_CLASSES
    def forward(self, lines = None, labels = None):
        """
        Forward function assumes lines is relative to model input dimensions.
        """
        out_labels = torch.zeros(self.num_classes)
        line_contents = {}
        if lines is not None:
            for idx, line in enumerate(lines):
                if labels[idx] not in line_contents:
                    line_contents[labels[idx]] = 0
                line_contents[labels[idx]] += (line[1] - line[0]) / self.length
                #Mark data as positive if more than threshold of it is positive.
                if line_contents[labels[idx]] > self.threshold:
                    out_labels[labels[idx]] = 1
        return out_labels
