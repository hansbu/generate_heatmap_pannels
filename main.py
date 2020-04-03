import collections
from utils import *
from FourPanelGleasonImage import FourPanelGleasonImage
from HeatMap import HeatMap
from StagedTumorHeatMap import StagedTumorHeatMap

# these folders will be replaced by paramaters
svs_fol = '/data01/tcga_data/tumor/luad'
staged_pred = '/data04/shared/hanle/quip_lung_cancer_detection_LUAD_TCGA/data/heatmap_txt_6classes_with_headers'
til_fol = '/data04/shared/shahira/TIL_heatmaps/LUAD/vgg_mix_binary/heatmap_txt'
output_pred = '4panel_pngs'

prefix = "prediction-"
wsi_extension = ".svs"
skip_first_line_pred = True

fns = [fn.split('prediction-')[-1] for fn in os.listdir(til_fol) if fn.startswith('prediction-') and not fn.endswith('low_res')]
til_wsiID_map = collections.defaultdict(str)
for fn in fns:
    til_wsiID_map[fn.split('.')[0]] = fn


def checkFileExisting(wsiId):
    #til_wsiID = til_wsiID_map[wsiId]  # if cancer id is different from til slide id
    til_wsiID = wsiId
    allPath = [
        os.path.join(staged_pred, 'color-' + wsiId), # colorPath
        os.path.join(svs_fol, wsiId + wsi_extension), # svsPath
        os.path.join(staged_pred, prefix + wsiId), # predPath
        os.path.join(til_fol, 'prediction-' + til_wsiID), # tilPath_pred
        os.path.join(til_fol, 'color-' + til_wsiID), # tilPath_color
    ]
    ans = True
    for path in allPath:
        ans = os.path.exists(path)
        if not ans:
            print(path, "does not exit!")
            break
    return ans


def gen1Image(fn):
    wsiId = fn[len(prefix):]
    if not checkFileExisting(wsiId):
        return

    oslide = openslide.OpenSlide(os.path.join(svs_fol, wsiId + wsi_extension))

    #til_wsiID = til_wsiID_map[wsiId]     # if cancer id is different from til slide id
    til_wsiID = wsiId
    til_heatmap = HeatMap(til_fol, skip_first_line_pred=False)
    til_heatmap.setWidthHeightByOSlide(oslide)
    til_map = til_heatmap.getHeatMapByID(til_wsiID)

    stagedCancerFile = StagedTumorHeatMap(staged_pred, skip_first_line_pred)
    stagedCancerFile.setWidthHeightByOSlide(oslide)
    stagedCancerMap = stagedCancerFile.getHeatMapByID(wsiId)
    classificationMap = stagedCancerFile.getStageClassificationTilMap(tilMap=til_map)
    # classificationMap = stagedCancerFile.getStageClassificationMap()    # no til file
    stageClassificationMap = stagedCancerFile.getStageClassificationMap()

    img = FourPanelGleasonImage(oslide, stagedCancerMap, classificationMap, stageClassificationMap,
                       os.path.join(output_pred, wsiId+".png"))
    img.saveImg()
    print(wsiId)


def main(parallel_processing = 0):
    if not os.path.isdir(output_pred):
        os.makedirs(output_pred)
    print('In main, prefix: ', prefix)

    grades_prediction_fns = [f for f in os.listdir(cancer_fol) if f.startswith(prefix) and not f.endswith('low_res')]

    if len(grades_prediction_fns) == 0:
        print("In main: No valid file!")
        return

    if parallel_processing in {0, 1}:
        for i, fn in enumerate(grades_prediction_fns):
            print(i, fn)
            gen1Image(fn)
    else:
        num_of_cores = multiprocessing.cpu_count() - 2
        if parallel_processing > 0:
            num_of_cores = min(num_of_cores, parallel_processing)

        print("Using multiprocessing, num_cores: ", num_of_cores)
        p = multiprocessing.Pool(num_of_cores)
        p.map(gen1Image, grades_prediction_fns)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        parallel_processing = 0
    else:
        parallel_processing = int(sys.argv[1])

    print("***********************************************")
    print("Usage: python main.py 0/1/4/-1")
    print("0/1: not using parallel processing")
    print("any number larger than 1, N, using N cores in parallel processing")
    print("-1: use all available cores in parallel processing, left 2 cores for others")
    print("***********************************************\n")

    main(parallel_processing = parallel_processing)



