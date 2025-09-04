import torch
import torchvision 
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from itertools import combinations
from ATLutils import simpleDataset,ITBDataLoader,Valid_Infernce
from ITBDOD import FasterRCNN


class ActiveLearning:
    '''
    'uncertainty', : a
     'class_difficulty', : b
     'class_ambiguity',: c
        'class_sparse', x
     'al_score'
    '''
    def __init__(self,TL_param,confidence,batch,img_foler_path):
        
        model = FasterRCNN.get_model_instance_segmentation()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        self.model=model.load_state_dict(torch.load(TL_param))
        self.device=device
        
        self.confidence=confidence
        self.batch=batch
        self.img_foler_path=img_foler_path #D:/OBJ_PAPER/Data/itb_p_500/


    def collate_fn(self):
        return tuple(zip(*self.batch))

    def get_transform(self):
        custom_transforms = []
        custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)
        
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
    
    def make_prediction(self,model, img, threshold):
        model.eval()
        preds = model(img)
        for id in range(len(preds)) :
            idx_list = []

            for idx, score in enumerate(preds[id]['scores']) :
                
                if score > threshold : 
                    idx_list.append(idx)

            preds[id]['boxes'] = preds[id]['boxes'][idx_list]
            preds[id]['labels'] = preds[id]['labels'][idx_list]
            preds[id]['scores'] = preds[id]['scores'][idx_list]

        return preds


    def img_uncertainty(self,model,train_data_loader,method,param):
        '''
        내림차순, 큰 수 부터
        ascend=False
        오름차순, 작은 수 부터
        ascend=True 
        
        '''
        self.a=param[0]
        self.b=param[1]
        self.c=param[2]

        img_id=[]
        pred_label=[]
        pred_scores=[]
        labels = []
        bbox=[]
        img_uncertianty=[]
        
        
        for im, annot in tqdm(train_data_loader, position = 0, leave = True):
            im = list(img.to(self.device) for img in im)
            #test data 이미지 id 저장
            img_id.append(annot[0]['image_id'].item())

            for t in annot:
                labels += t['labels']

            with torch.no_grad():
                preds_adj = self.make_prediction(model, im, self.confidence)
                preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]

                #예측 label 저장
                pred_label.append(preds_adj[0]['labels'].tolist())
                #box
                bbox.append((preds_adj[0]['boxes'].tolist()))
                #예측 score 저장
                pred_scores.append(preds_adj[0]['scores'].tolist())
                img_uncertianty.append(np.mean(preds_adj[0]['scores'].tolist()))

        model_pred=pd.DataFrame(img_id,columns=['img_id'])
        model_pred['label']=pred_label
        model_pred['score']=pred_scores
        model_pred['bbox']=bbox
        #confidence가 0.5이상인 객체의 score 합
        uncertainty=[]
        for i in range(len(model_pred)):

            scores=model_pred.iloc[i]['score']
            un=0
            for s in scores:

                if s>self.a:
                    un=un+s
            uncertainty.append(un)

        model_pred['uncertainty']=uncertainty
        
        #confidence가 0.3이하인 객체의 개수
        class_difficulty=[]
        for i in range(len(model_pred)):
            cls_dif=0
            scores=model_pred.iloc[i]['score']
            for s in scores:
                if s<self.b:
                    cls_dif+=1
            class_difficulty.append(cls_dif)

        model_pred['class_difficulty']=class_difficulty
        
        #iou 계산하여 클래스가 다르면서 중첩인 box개수
        class_ambiguity=[]
        for a in range(len(model_pred)):
            box_index=list(range(len(model_pred.iloc[a]['bbox'])))
            label_index=list(range(len(model_pred.iloc[a]['label'])))
            iou_index=list(combinations(box_index, 2))

            cls_amb=0
            for i in iou_index:
                box1_i=i[0]
                box2_i=i[1]

                #box좌표
                box1=model_pred.iloc[a]['bbox'][box1_i]
                box2=model_pred.iloc[a]['bbox'][box2_i]

                #라벨
                box1_label=model_pred.iloc[a]['label'][box1_i]
                box2_label=model_pred.iloc[a]['label'][box2_i]

                #class가 다른 경우

                if (box1_label!= box2_label):
                    #iou 계산
                    iou=self.bbox_iou(box1, box2, x1y1x2y2=True)
                    if iou>self.c:
                        cls_amb+=1
            class_ambiguity.append(cls_amb)        
                #클래스가 같은경우

        model_pred['class_ambiguity']=class_ambiguity
        
        #class sparse, 예측된 라벨 중 특정 라벨의 수가 적은 경우 해당 라벨을 우선적으로 선택
        text=0
        form=0
        table=0

        for i in range(len(model_pred)):

            label_line=model_pred['label'].iloc[i]
            for l in label_line:
                if l==1:
                    form+=1

                elif l==2:
                    table+=1

                else:
                    text+=1

        total=text+form+table
        text_ratio=round(text/total,5)
        form_ratio=round(form/total,5)
        table_ratio=round(table/total,5)
        tt=[text_ratio,form_ratio,table_ratio]
        total_ratio=pd.DataFrame(tt,columns=['label'])
        total_ratio.index=['text_spars','form_spars','table_spars']
        #비율이 가장 낮은 label
        spars=total_ratio.sort_values(by=['label'],ascending=True).index[0]

        text_spars=[]
        form_spars=[]
        table_spars=[]

        for i in range(len(model_pred)):

            img_text=0
            img_form=0
            img_table=0

            label_line=model_pred['label'].iloc[i]
            for l in label_line:
                if l==1:
                    img_form+=1
                elif l==2:
                    img_table+=1
                else:
                    img_text+=1

            text_spars.append(img_text*text_ratio)
            form_spars.append(img_form*form_ratio)
            table_spars.append(img_table*table_ratio)

        model_pred['text_spars']=text_spars
        model_pred['form_spars']=form_spars
        model_pred['table_spars']=table_spars
        
        model_pred['class_sparse']=model_pred[spars]
        model_pred['al_score']=model_pred[spars]+model_pred['uncertainty']+model_pred['class_difficulty']+model_pred['class_ambiguity']
        model_pred.sort_values(by=[method],ascending=False,inplace=True)
             
        return model_pred


    def sampling(self,train,train_data_loader,method,param):
        model_pred=self.img_uncertainty(self.model,train_data_loader,method,param)
        sample_idx=model_pred[:self.batch]['img_id'].tolist()
        sample=train.loc[sample_idx]
        train_rest=train.drop(sample.index)
        train_rest_dataset = simpleDataset(dataset=train_rest,
                              resize=4,
                              color='L',
                              img_foler_path=self.img_foler_path,
                              transforms=self.get_transform())

        train_rest_data_loader = torch.utils.data.DataLoader(train_rest_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=self.collate_fn)
      
        return model_pred,sample,train_rest,train_rest_data_loader
    
    def train_AL(self,total,method,param,param_name,TL_param,result_path,save=False,save_param_path=None):  
        path=self.img_foler_path
        num_epochs = 100
        patience=10
        
        num_classes = 4
        train_layer=5
        model = FasterRCNN.get_model_instance_segmentation(num_classes,train_layer)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        model.load_state_dict(torch.load(TL_param))

        # parameters
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.0001)
        
        iteration = 0    

        Sampled_img=pd.DataFrame()
        
        #data reload
        data=ITBDataLoader(path,self.batch)
        test,valid,train,train_data_loader,valid_data_loader, test_data_loader=data.train_valid_test_split(train)
        
        while len(Sampled_img)<total:

            if len(Sampled_img)+self.batch>total:
                batch=(len(Sampled_img)+self.batch)-total
                sample=train.sample(n=batch)
                Sampled_img=Sampled_img.append(sample,ignore_index=True)
                sample_data_loader=data.data_load(Sampled_img)
                
            else:

                model_pred,sampled,train_rest,train_rest_data_loader=self.sampling(train,train_data_loader,method,param)

                Sampled_img=Sampled_img.append(sampled,ignore_index=True)  
                sample_data_loader=data.data_load(Sampled_img)
                
                #sample 제외한 나머지 train 데이터
                train=train_rest

            iteration += 1
            print(iteration)
            model.train()
            not_save_count=0
            
            #평균 loss
            avg_train_loss=[]
            avg_valid_loss=[]
            
            for epoch in range(num_epochs):

                # 모델이 학습되는 동안 trainning loss를 track
                train_losses = []
                # 모델이 학습되는 동안 validation loss를 track
                valid_losses = []
                
                st = time.time()
                for imgs, annotations in sample_data_loader:

                    imgs = list(img.to(device) for img in imgs)
                    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
                    loss_dict = model(imgs, annotations)
                   
                    losses = sum(loss for loss in loss_dict.values())
                    train_losses.append(losses.item())       
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                #이미지 한장당 평균 loss
                avg_train_loss.append(np.mean(train_losses).round(5))

                #validation, early_stop, save weights
                with torch.no_grad():

                    for im, annot in valid_data_loader:
                        im = list(img.to(device) for img in im)
                        annot = [{k: v.to(device) for k, v in t.items()} for t in annot]
                        val_loss_dict = model(im, annot)
                        val_losses = sum(val_loss for val_loss in val_loss_dict.values())
                        valid_losses.append(val_losses.item())

                    epoch_val_loss=np.mean(valid_losses).round(5)
                    avg_valid_loss.append(epoch_val_loss)  

                fi = time.time()     
                print('epoch:',epoch,'train_loss: ',np.mean(train_losses).round(5),'validation loss : ',np.mean(valid_losses).round(5),'time',fi-st)

                min_val_loss=np.min(avg_valid_loss)
                if min_val_loss>=epoch_val_loss:

                    torch.save(model.state_dict(),save_param_path)
                    not_save_count=0
                    print('epoch:',epoch,'save model')

                else:
                    not_save_count+=1
                    model.load_state_dict(torch.load(save_param_path))
                    if not_save_count>=patience:
                        print('no more training')
                        break

            fi = time.time()     
            print('iteration:',iteration,'train_loss: ',np.mean(train_losses).round(5),'time',fi-st)
            print('sample num:',len(Sampled_img))

            VI=Valid_Infernce(model,save_param_path,test_data_loader)
            VI.valid(0.5,iteration,save=True,save_path=result_path.format(total,method,param_name))

        return 0