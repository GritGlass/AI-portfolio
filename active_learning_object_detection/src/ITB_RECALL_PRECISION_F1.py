import torch
import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

class Precision_Recall_F1:
    
    def __init__(self,iou_threds,preds_adj_all,annot_all):
        
        self.iou_threds=iou_threds
        self.preds_adj_all=preds_adj_all
        self.annot_all=annot_all
    

    def bbox_iou(self,box1, box2, x1y1x2y2=True):
        """
        Returns the IoU of two bounding boxes
        """
        box1=torch.tensor(box1)
        box2=torch.tensor(box2)

        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou



    def TRUE_POSITIVE(self): 
        '''
        1: form
        2: table
        3: text
        Recall=[form, table, text]
        '''
        
        #i=이미지 number, TP구하기
        TP_form=0
        TP_table=0
        TP_text=0

        for i in range(len(self.annot_all)):
            for t in range(len(self.annot_all[i][0]['boxes'])):
        
                
                for p in range(len(self.preds_adj_all[i][0]['boxes'])):
                   
                    
                    #iou 계산
                    box1=self.preds_adj_all[i][0]['boxes'][p]
                    box2=self.annot_all[i][0]['boxes'][t]
                    iou=self.bbox_iou(box1, box2, x1y1x2y2=True)

                    if iou>= self.iou_threds:
                        #라벨 일치하는지 계산
                        pred_label=self.preds_adj_all[i][0]['labels'][p].item()
                        gt=self.annot_all[i][0]['labels'][t].item()
                        if pred_label==gt:

                            if gt==1:
                                TP_form+=1
                                break
                        
                            elif gt==2:
                                TP_table+=1
                                break
                             
                            else:
                                TP_text+=1
                                break
                                
                        else:
                           
                            continue

   
        return TP_form,TP_table,TP_text
         
    def PRECISION(self):
        TP_form,TP_table,TP_text=self.TRUE_POSITIVE()
        
        #모델이 예측한 전체 라벨 값
        Pred_form=0+sys.float_info.epsilon
        Pred_table=0+sys.float_info.epsilon
        Pred_text=0+sys.float_info.epsilon
        
        for a in range(len(self.preds_adj_all)):
            label=self.preds_adj_all[a][0]['labels'].tolist()
            for i in label:
                if i==1:
                    Pred_form+=1
                elif i==2:
                    Pred_table+=1
                else:
                    Pred_text+=1

        precision_form=round(TP_form/Pred_form,5)
        precision_table=round(TP_table/Pred_table,5)
        precision_text=round(TP_text/Pred_text,5)

        Precision=[precision_form,precision_table,precision_text]
        mean_p=round(np.mean(Precision),5)
        return Precision,mean_p
    
    
    def RECALL(self): 
        TP_form,TP_table,TP_text=self.TRUE_POSITIVE()
        
        GT_form=0+sys.float_info.epsilon
        GT_table=0+sys.float_info.epsilon
        GT_text=0+sys.float_info.epsilon
        
        for a in range(len(self.annot_all)):
            label=self.annot_all[a][0]['labels'].tolist()
            for i in label:
                if i==1:
                    GT_form+=1
                elif i==2:
                    GT_table+=1
                else:
                    GT_text+=1


        recall_form=round(TP_form/GT_form,5)
        recall_table=round(TP_table/GT_table,5)
        recall_text=round(TP_text/GT_text,5)

        Recall=[recall_form,recall_table,recall_text]
        mean_r=round(np.mean(Recall),5)
        return Recall,mean_r
    
    
    def F1_SCORE(self):
        precision,mean_p=self.PRECISION()
        recall,mean_r=self.RECALL()
        print('precision',precision)
        print('recall',recall)
        
        
        #form f1_score
        form_f1=2*(precision[0]*recall[0])/(precision[0]+recall[0]+sys.float_info.epsilon)
        
        #table f1_score
        table_f1=2*(precision[1]*recall[1])/(precision[1]+recall[1]+sys.float_info.epsilon)
        
        #text f1_score
        text_f1=2*(precision[2]*recall[2])/(precision[2]+recall[2]+sys.float_info.epsilon)
        
        f1_score=[round(form_f1,5),round(table_f1,5),round(text_f1,5)]
        
        #average f1-score
        avg_f1=2*mean_p*mean_r/(mean_p+mean_r+sys.float_info.epsilon)
        
        return f1_score,avg_f1
 
   