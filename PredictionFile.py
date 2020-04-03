from XYLabelFile import XYLabelFile
import numpy as np


class PredictionFile(XYLabelFile):
    def __init__(self, file_path, skip_header=False):
        super().__init__(file_path, skip_header)
        self.pred = None
        self.necr = None

    def get_pred_and_necr(self):
        if self.data.shape[1] > 3:
            self.pred, self.necr = self.extract([2, 3])
        else:
            self.pred = self.extract([2])
            self.necr = np.zeros_like(self.pred, dtype=self.pred.dtype)
        return self.pred, self.necr, self.patchSize

    def get_labeled_im(self):
        return self.get_pred_and_necr()
