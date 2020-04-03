from utils import *

class XYLabelFile(object):
    """
    This class reads cvs file seperated by ' ', every line of which is like x y ...
    And convert it to x,y 2D array.
    The whole process is: 1)init 2)set width and height 3)extract
    """
    def __init__(self, file_path, skip_header=False):
        """ Read the file. """
        self.filePath = file_path
        self.skip_header = skip_header
        self.data = self.read_file(self.filePath)
        self.patchSize = 0
        self.width = 0
        self.height = 0
        self.extracted = None

    def setWidthHeightByOSlide(self, slide):
        self.width = slide.dimensions[0]
        self.height = slide.dimensions[1]

    def setWidthHeight(self, width, height):
        self.width = width
        self.height = height

    def read_file(self, file_path):
        """ Read prediction file into a numpy 2D array """
        return np.genfromtxt(file_path, delimiter=' ', skip_header=1 if self.skip_header else 0)

    def extract(self, indexes):
        """
        This function extracts the column of csv file you want.
        Args:
            indexes(list): The indexes of columns you want to extracted
        Retures:
            list: self.extracted. List of extracted 2D arrays.
        """
        uniqueX = np.unique(self.data[:, 0])
        uniqueX.sort()
        self.patchSize = uniqueX[5] - uniqueX[4]
        #print('\nfile: ', self.filePath)
        #print('uniqueX: ', uniqueX[:5])
        #print("max x,y:", np.max(self.data[:, [0,1]], axis=0))
        #print('patch size : ', self.patchSize)
        rowIndex = np.logical_and(self.data[:,0] + self.patchSize / 2 < self.width,
                                  self.data[:,1] + self.patchSize / 2 < self.height)
        filteredData = self.data[rowIndex, :]

        xys_pred = ((filteredData[:, [0, 1]] + self.patchSize / 2) / self.patchSize - 1).astype(np.int)
        #print("oslide width & height:", self.width, self.height)
        #print("filteredData.shape", filteredData.shape)
        #print("data.shape:", self.data.shape)
        #print("index min x,y:", np.min(xys_pred, axis=0))
        #print("index max x,y:", np.max(xys_pred, axis=0))
        self.extracted = []
        shape = int(self.width // self.patchSize), int(self.height // self.patchSize)
#         print("small shape:", shape)
        for i in indexes:
            self.extracted.append(np.zeros(shape))

        for i in range(xys_pred.shape[0]):
            l = filteredData[i,:]
            x, y = xys_pred[i]
            for j, arr in zip(indexes, self.extracted):
                arr[x,y] = l[j]
        return self.extracted

