from XYLabelFile import XYLabelFile
import numpy as np
import pdb

class StagePredictionFile(XYLabelFile):
    def __init__(self, file_path, skip_header=False):
        super().__init__(file_path, skip_header)
        self.lepidic = None
        self.benign = None
        self.acinar = None
        self.micropap = None
        self.mucinous = None
        self.solid = None
        self.pred = None
        self.benign_adjusted = None

    def get_stage_prediction(self):
        indexes = [2, 3, 4, 5, 6, 7]
        self.lepidic, self.benign, self.acinar, self.micropap, self.mucinous, self.solid = self.extract(indexes)
        sum_prob = self.lepidic + self.benign + self.acinar + self.micropap + self.mucinous + self.solid
        self.benign_adjusted, self.pred = self.get_benign_tumor_adjusted(self.benign, sum_prob, len(indexes))
        return self.lepidic, self.benign, self.acinar, self.micropap, self.mucinous, self.solid

    def get_benign_tumor_adjusted(self, benign, sum_probabilities, num_classes):
        # benign, sum_probabilities is a matrix w x h
        # num_classes is number of classes in the classification network
        w, h = benign.shape
        benign_adjusted = np.zeros(w * h)
        tumor_adjusted = np.zeros(w * h)
        sum_probabilities = sum_probabilities.reshape(-1)
        benign_ = benign.reshape(-1)
        for i, p in enumerate(benign_):
            if sum_probabilities[i] > 0:
                benign_adjusted[i] = benign_[i] * (num_classes - 1) / (sum_probabilities[i] + benign_[i] * (num_classes - 2))
                tumor_adjusted[i] = 1 - benign_adjusted[i]
        return benign_adjusted.reshape(w, h), tumor_adjusted.reshape(w, h)

    def get_labeled_im(self):
        return self.get_stage_prediction()
