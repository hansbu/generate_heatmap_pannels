from XYLabelFile import XYLabelFile


class ColorFile(XYLabelFile):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.whiteness = None
        self.blackness = None
        self.redness = None

    def get_color_channels(self):
        self.whiteness, self.blackness, self.redness = self.extract([2, 3, 4])
        return self.whiteness, self.blackness, self.redness

    def get_whiteness_im(self):
        return self.get_color_channels()
