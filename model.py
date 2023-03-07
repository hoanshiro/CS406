import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def FasterRCNN(num_classes, pretrained=False):
    # 7 classes + 1 background
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
    return model


def RetinaNet(num_classes, pretrained=False):
    retina = torchvision.models.detection.retinanet_resnet50_fpn(num_classes, pretrained, pretrained_backbone = True)
    return retina
    
