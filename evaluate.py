import torch
import utils_ObjectDetection as utils
from tqdm import tqdm
from inference import make_prediction


def calculate_mAP(model, test_loader, device):
    labels = []
    preds_adj_all = []
    annot_all = []

    for idx, (im, annot) in enumerate(tqdm(test_loader, position=0, leave=True)):
        im = list(img.to(device) for img in im)
        if idx == 10:
            break
        for t in annot:
            labels += t['labels']

        with torch.no_grad():
            preds_adj = make_prediction(model, im, 0.05)
            preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(annot)
    # iou 0.3, 0.05
    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += utils.get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5)

    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in
                                                list(zip(*sample_metrics))]  # all the batches get concatenated
    precision, recall, AP, f1, ap_class = utils.ap_per_class(true_positives, pred_scores, pred_labels,
                                                             torch.tensor(labels))
    mAP = torch.mean(AP)
    print("\n-----Metric-----\n")
    print(f'mAP : {mAP}')
    print(f'AP : {AP}')
    return mAP
