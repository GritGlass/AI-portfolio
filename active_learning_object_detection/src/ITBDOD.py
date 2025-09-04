import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN:
    def __init__(self,num_classes=4, train_layers=3,pretrain=True):
        self.num_classes=num_classes
        self.train_layers=train_layers
        self.pretrain=pretrain

    def get_model_instance_segmentation(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=self.pretrain,trainable_backbone_layers=self.train_layer)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model