import json
import argparse
import os
import glob
import logging
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("COCOEval")
    
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser("Evaluate a COCO prediction file")
    parser.add_argument("--coco_path", type=str, help="Path to COCO 2017 Val folder", required=True)
    parser.add_argument("--pred_path", type=str, help="Path to prediction json", required=True)
    parser.add_argument("--gt_path", type=str, help="Path to ground truth json", required=True)
    
    args = parser.parse_args()
      
    coco_glob = os.path.join(args.coco_path, "*.jpg")
    images = glob.glob(coco_glob)
    
    logger.info("Looking for: {}".format(coco_glob))
    ids = [int(os.path.basename(i).split('.')[0]) for i in images]
        
    
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    anno = COCO(args.gt_path)  # init annotations api
    pred = anno.loadRes(args.pred_path)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.params.imgIds = ids
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    
    logger.info("mAP: {}".format(map))
    logger.info("mAP50: {}".format(map50))
