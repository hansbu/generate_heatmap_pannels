from ColorFile import ColorFile
from PredictionFile import PredictionFile
from utils import *


class HeatMap(object):

    def __init__(self, rootFolder, skip_first_line_pred):
        self.rootFolder = rootFolder
        self.width = 0
        self.height = 0
        self.skip_first_line_pred = skip_first_line_pred

    def setWidthHeightByOSlide(self, slide):
        self.width = slide.dimensions[0]
        self.height = slide.dimensions[1]

    def getHeatMapByID(self, slideId, prefix = 'prediction-'):
        """
        Args:
            slideId(str): id of svs file, like 'TCGA-3C-AALI-01Z-00-DX1'.
        """
        predictionFileName = prefix + slideId
        colorFileName = 'color-' + slideId
        self.heatmap = self.getHeatMap(os.path.join(self.rootFolder, predictionFileName),
                               os.path.join(self.rootFolder, colorFileName))
        return self.heatmap

    def getHeatMap(self, predPath, colorPath):
        """
        Args:
            predPath(str): must be full path.
            colorPath(str): must be full path.
        """
        predictionFile = PredictionFile(predPath, self.skip_first_line_pred)
        predictionFile.setWidthHeight(self.width, self.height)
        pred, necr, patch_size = predictionFile.get_pred_and_necr()
#         print("pred.shape:",pred.shape)

        colorFile = ColorFile(colorPath)
        colorFile.setWidthHeight(self.width, self.height)
        whiteness, blackness, redness = colorFile.get_whiteness_im()

        image = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

        image[:, :, 0] = np.multiply(255* pred, (blackness>30).astype(np.float64), (redness<0.15).astype(np.float64))
        image[:, :, 1] = necr
        image[:, :, 2] = 255 * (cv2.GaussianBlur(whiteness, (5, 5), 0) > 12)
        out = image[:,:, [2, 1, 0]]
        out = np.transpose(out, (1, 0, 2))
        return out