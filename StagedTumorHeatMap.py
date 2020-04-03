from ColorFile import ColorFile
from HeatMap import HeatMap
from StagePredictionFile import StagePredictionFile
from utils import *


class StagedTumorHeatMap(HeatMap):

    def __init__(self, rootFolder, skip_first_line_pred):
        super().__init__(rootFolder, skip_first_line_pred)

    def getHeatMap(self, stagePredPath, colorPath):
        predictionFile = StagePredictionFile(stagePredPath, self.skip_first_line_pred)
        predictionFile.setWidthHeight(self.width, self.height)
        predictionFile.get_stage_prediction()

        pred = predictionFile.pred
        # print("StagedTumorPred.shape:",pred.shape)

        colorFile = ColorFile(colorPath)
        colorFile.setWidthHeight(self.width, self.height)
        whiteness, blackness, redness = colorFile.get_whiteness_im()

        image = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

        image[:, :, 0] = np.multiply(255 * pred, (blackness > 30).astype(np.float32),
                                     (redness < 0.15).astype(np.float32))

        image[:, :, 2] = 255 * (cv2.GaussianBlur(whiteness, (5, 5), 0) > 12)
        out = image[:, :, [2, 1, 0]]
        out = np.transpose(out, (1, 0, 2))

        self.predictionFile = predictionFile
        self.colorFile = colorFile
        self.tissue = image[:, :, 2]
        return out

    def getTumorClassificationMap(self):
        predictionFile = self.predictionFile
        stackedArray = np.stack([predictionFile.pred,
                                 predictionFile.benign_adjusted],
                                axis=2)
        classification = np.argmax(stackedArray, axis=2)
        mask = np.sum(stackedArray, axis=2) > 0.1

        colorArray = np.array([  # rgb array
            [255, 255, 0],  # tumor yellow
            [0, 0, 255]  # benign blue
        ])
        self.tumorColorArray = colorArray

        image = np.ones((stackedArray.shape[0], stackedArray.shape[1], 3), dtype=np.uint8)
        image = image * 192  # all gray
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if self.tissue[i, j] > 100 and mask[i, j]:
                    image[i, j] = colorArray[classification[i, j], :]

        image = np.transpose(image, (1, 0, 2))
        image = image[:, :, [2, 1, 0]]  # convert bgr
        self.tumorClassificationMap = image
        return image

    def getStageClassificationMap(self):
        predictionFile = self.predictionFile
        stackedArray = np.stack([predictionFile.lepidic,
                                 predictionFile.benign,
                                 predictionFile.acinar,
                                 predictionFile.micropap,
                                 predictionFile.mucinous,
                                 predictionFile.solid],  # stroma
                                axis=2)

        classification = np.argmax(stackedArray, axis=2)
        mask = np.sum(stackedArray, axis=2) > 0.05

        colorArray = np.array([  # rgb array
            [255, 0, 0],  # red
            [0, 0, 255],  # benign blue
            [255, 127, 0],# orange
            [255, 255, 0],# yellow
            [0, 255, 0],  # green
            [139, 0, 255] # violet
        ])
        self.stageColorArray = colorArray

        image = np.ones((stackedArray.shape[0], stackedArray.shape[1], 3), dtype=np.uint8)
        image = image * 192  # all gray
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if self.tissue[i, j] > 100 and mask[i, j]:
                    image[i, j] = colorArray[classification[i, j], :]

        image = np.transpose(image, (1, 0, 2))
        image = image[:, :, [2, 1, 0]]  # convert bgr
        self.stageClassificationMap = image
        return image

    def thresholding(self, array):
        threshold = 255 * 0.5
        out = np.zeros_like(array, dtype=np.uint8)
        out[array > threshold] = 255
        return out

    def getStageClassificationTilMap(self, tilMap):
        cancerProb = self.predictionFile.pred
        cancerProb = np.transpose(cancerProb, (1, 0))
        lympImg = tilMap
        cancerArray = self.thresholding(cancerProb * 255)

        up = int(math.ceil(lympImg.shape[0] / cancerArray.shape[0]))

        if up > 1:
            iml_u = np.zeros((cancerArray.shape[0] * up, cancerArray.shape[1] * up), dtype=np.float32)
            for x in range(cancerArray.shape[1]):
                for y in range(cancerArray.shape[0]):
                    iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = cancerArray[y, x]

            cancerArray = iml_u.astype(np.uint8)

        smooth5 = cancerArray
        if np.max(smooth5) < 2:
            smooth5 = (smooth5 * 255).astype(np.uint8)

        smooth5 = cv2.resize(smooth5, (lympImg.shape[1], lympImg.shape[0]), interpolation=cv2.INTER_LINEAR)
        smooth5 = cv2.GaussianBlur(smooth5, (5, 5), 0)

        out = np.zeros_like(lympImg, dtype=np.uint8)

        for i in range(lympImg.shape[0]):
            for j in range(lympImg.shape[1]):
                b, g, r = lympImg[i, j]
                out[i, j] = np.array([192, 192, 192])
                is_tumor = smooth5[i, j] > 100
                is_lym = r > 100
                is_tisue = (b >= 0.5 * 255)
                # Tissue, Tumor, Lym
                if (not is_tumor) and (not is_lym):  # BGR
                    if not is_tisue:
                        out[i, j] = np.array([255, 255, 255])  # White
                    else:
                        out[i, j] = np.array([255, 0, 0])  # blue # original 192 gray
                elif is_tumor and (not is_lym):
                    # print("tumor & not lym")
                    out[i, j] = np.array([0, 255, 255])  # Yellow
                elif (not is_tumor) and is_lym:
                    out[i, j] = np.array([0, 0, 200])  # Redish
                else:
                    out[i, j] = np.array([0, 0, 255])  # Red
        return out
