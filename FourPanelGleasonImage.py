from utils import  *


class FourPanelGleasonImage(object):
    def __init__(self, oslide, cancerImg, classificationMap, stageClassificationMap, savePath):
        self.oslide = oslide
        self.cancerImg = cancerImg
        self.classificationMap = classificationMap
        self.stageClassificationMap = stageClassificationMap
        self.savePath = savePath

    def saveImg(self):
        shape = (self.cancerImg.shape[1] * 2, self.cancerImg.shape[0] * 2)
        thumbnail = self.oslide.get_thumbnail(shape)

        cancerImg = self.cancerImg

        classificationMap = self.classificationMap[:, :, [2, 1, 0]]  # convert to rgb
        mergedMap = self.stageClassificationMap[:, :, [2, 1, 0]]

        cancerSmoothImg = cv2.GaussianBlur(cancerImg, (5, 5), 0)

        aspect = cancerImg.shape[0] / cancerImg.shape[1]

        width = 6.4 * 1
        mpl.rcParams["figure.figsize"] = [width * 1.10, width * aspect]
        mpl.rcParams["figure.dpi"] = 600
        #         print(aspect)

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
                # axarr[x, y].axis('off')
                # axarr[x, y].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                # axarr[x, y].set_aspect(lymImg.shape[0] / lymImg.shape[1])

                # axarr[x, y].axis('off')
                axarr[x, y].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                caxarr[x, y].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                caxarr[x, y].axis("off")


        axarr[0, 0].imshow(thumbnail)


        axarr[1, 0].imshow(mergedMap)
        colors_merge = ['#00FF00', 'orange', 'blue']  # #00FF00 for pure green
        labels_merge = ['G3', 'G4+5', 'Benign']
        legend_patches_merge = [Rectangle((0, 0), width=0.01, height=0.01, color=icolor, label=label, lw=0)
                                for icolor, label in zip(colors_merge, labels_merge)]
        caxarr[1, 0].legend(handles=legend_patches_merge,
                            facecolor=None,  # "white",
                            edgecolor="white",
                            fancybox=None,
                            bbox_to_anchor=(-0.1, 0),
                            loc='lower left',
                            fontsize='x-small',
                            borderpad=0)


        axarr[1, 1].imshow(classificationMap)
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

        #         lymImg = classificationMap
        #         lymR = lymImg[:, :, 0]  # bgr? rgb
        #         lymInts = lymR.astype(np.float)
        #         lymInts = lymInts / 255

        #        lymB = lymImg[:, :, 2]  # b
        #        lymInts[lymB < 100] = None
        #        lymIm = axarr[1, 0].imshow(lymInts, cmap='jet', vmax=1.0, vmin=0.0)


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

        plt.savefig(self.savePath, bbox_inches='tight')
        plt.close()