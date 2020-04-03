from utils import *



class MergedHeatMap(object):
    """
    Merge cancer heatmap and lymp heatmap together.
    Input image arrays should be opencv type(BGR).
    """
    def __init__(self, cancerImg, lympImg):
        self.cancerImg = cancerImg
        self.lympImg = lympImg
        self.mergedHeatMap = self.merge()

    def thresholding(self, array):
        threshold = 255 * 0.5
        out = np.zeros_like(array, dtype=np.uint8)
        out[array > threshold] = 255
        return out

    def merge(self):
        cancerImg, lympImg = self.cancerImg, self.lympImg
        cancerArray = self.thresholding(cancerImg[:, :, 2])

        up = int(math.ceil(lympImg.shape[0]/cancerArray.shape[0]))

        if up > 1:
            iml_u = np.zeros((cancerArray.shape[0] * up, cancerArray.shape[1] * up), dtype=np.float32)
            for x in range(cancerArray.shape[1]):
                for y in range(cancerArray.shape[0]):
                    iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = cancerArray[y, x]

            cancerArray = iml_u.astype(np.uint8)

        smooth5 = cancerArray
        if np.max(smooth5) < 2:
            smooth5 = (smooth5*255).astype(np.uint8)

        smooth5 = cv2.resize(smooth5, (lympImg.shape[1], lympImg.shape[0]), interpolation=cv2.INTER_LINEAR)
        smooth5 = cv2.GaussianBlur(smooth5, (5, 5), 0)

        out = np.zeros_like(lympImg, dtype=np.uint8)

        for i in range(lympImg.shape[0]):
            for j in range(lympImg.shape[1]):
                b, g, r = lympImg[i, j]
                out[i, j] = np.array([192,192,192])
                is_tumor = smooth5[i, j] > 100
                is_lym = r > 100
                is_tisue = (b >= 0.5 * 255)
                # Tissue, Tumor, Lym
                if (not is_tumor) and (not is_lym): # BGR
                    if not is_tisue:
                        out[i, j] = np.array([255,255,255]) #White
                    else:
                        out[i, j] = np.array([192,192,192]) # Grey
                elif is_tumor and (not is_lym):
                    out[i, j] = np.array([0,255,255]) #Yellow
                elif (not is_tumor) and is_lym:
                    out[i, j] = np.array([0,0,200]) #Redish
                else:
                    out[i, j] = np.array([0,0,255]) #Red
        return out
