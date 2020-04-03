from MergedHeatMap import MergedHeatMap
from utils import  *


class FourPanelImage(object):
    def __init__(self, oslide, cancerMap, tilMap, savePath):
        self.oslide = oslide
        self.cancerMap = cancerMap
        self.tilMap = tilMap
        self.savePath = savePath

    def saveImg(self):
        shape = (self.cancerMap.shape[1] * 2, self.cancerMap.shape[0] * 2)
        thumbnail = self.oslide.get_thumbnail(shape)

        mergedMap = MergedHeatMap(self.cancerMap, self.tilMap).mergedHeatMap
        mergedMap = mergedMap[:, :, [2, 1, 0]]  # convert to rgb

        cancerSmoothImg = cv2.GaussianBlur(self.cancerMap, (5, 5), 0)
        TilMapSmoothImg = cv2.GaussianBlur(self.tilMap, (5, 5), 0)

        aspect = self.cancerMap.shape[0] / self.cancerMap.shape[1]

        width = 6.4 * 1
        mpl.rcParams["figure.figsize"] = [width * 1.10, width * aspect]
        mpl.rcParams["figure.dpi"] = 300

        if aspect > 1:
            hspace = 0.04
            wspace = hspace * aspect
        else:
            wspace = 0.04
            hspace = wspace / aspect  # / aspect
        # * aspect
        fig2, axarr = plt.subplots(2, 2, gridspec_kw={'wspace': 0.28, 'hspace': hspace})

        caxarr = []
        for r in range(2):
            for c in range(2):
                divider = make_axes_locatable(axarr[r, c])
                cax = divider.append_axes("right", size="5%", pad=0)
                caxarr.append(cax)
        caxarr = np.array(caxarr).reshape(2, 2)

        for x in [0, 1]:
            for y in [0, 1]:
                axarr[x, y].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                caxarr[x, y].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                caxarr[x, y].axis("off")


        axarr[0, 0].imshow(thumbnail)

        # heat map of cancer
        cancerR = cancerSmoothImg[:, :, 2]  # bgr
        cancerInts = cancerR.astype(np.float)
        cancerInts = cancerInts / 255
        cancerB = cancerSmoothImg[:, :, 0]  # b
        cancerInts[cancerB < 100] = None
        cancerIm = axarr[0, 1].imshow(cancerInts, cmap='jet', vmax=1.0, vmin=0.0)

        axins = inset_axes(caxarr[0, 1],
                           width="50%",  # width = 5% of parent_bbox width
                           height="60%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(0.2, 0, 1, 1),
                           bbox_transform=caxarr[0, 1].transAxes,
                           borderpad=0,
                           )

        cb = fig2.colorbar(cancerIm, cax=axins)
        cb.ax.tick_params(labelsize='xx-small')
        caxarr[0, 1].axis("off")


        # heat map of TIL
        TilR = TilMapSmoothImg[:, :, 2]  # bgr
        TilInts = TilR.astype(np.float)
        TilInts = TilInts / 255
        TilB = TilMapSmoothImg[:, :, 0]  # b
        TilInts[TilB < 50] = None
        TilIm = axarr[1, 0].imshow(TilInts, cmap='jet', vmax=1.0, vmin=0.0)

        axins = inset_axes(caxarr[1, 0],
                           width="50%",  # width = 5% of parent_bbox width
                           height="60%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(0.2, 0, 1, 1),
                           bbox_transform=caxarr[1, 0].transAxes,
                           borderpad=0,
                           )
        cb = fig2.colorbar(TilIm, cax=axins)
        cb.ax.tick_params(labelsize='xx-small')
        caxarr[1, 0].axis("off")


        # megered map
        axarr[1, 1].imshow(mergedMap)
        colors_classification = ['red', 'yellow', 'blue']
        labels_classification = ['L', 'C', 'T']
        legend_patches_classification = [Rectangle((0, 0), width=0.01, height=0.01, color=icolor, label=label, lw=0)
                                         for icolor, label in zip(colors_classification, labels_classification)]
        caxarr[1, 1].legend(handles=legend_patches_classification,
                            facecolor=None,  # "white",
                            edgecolor=None,
                            fancybox=False,
                            bbox_to_anchor=(-0.1, 0),
                            loc='lower left',
                            fontsize='x-small',
                            shadow=False,
                            framealpha=0.,
                            borderpad=0)

        plt.savefig(self.savePath, bbox_inches='tight')
        plt.close()