import torch
import torchvision 
import pandas as pd
from PIL import Image
from tqdm import tqdm
import Performance_Measures as rp_f1


class simpleDataset(object):
    def __init__(self,dataset,resize,color, img_foler_path,transforms=None):
            self.transforms = transforms
            self.adnoc = dataset
            self.ids = dataset.index
            self. filenames=dataset['path'].to_list()
            self.resize =resize
            self.color=color
            self.img_foler_path=img_foler_path
            
    def __getitem__(self, index):
        adnoc_df = self.adnoc
        img_id = self.ids[index]
       # List: get annotation id from coco, index번째 annotation가져오기 [[1,2,3,4],[5,6,7,8]]
        annotation = adnoc_df['box'][img_id]

        # open the input image, 흑백=L , 단색=1
        img= Image.open(str(self.img_foler_path)+str(adnoc_df.loc[img_id]['path'])).convert(self.color)
        img= img.resize((int(img.width / self.resize), int(img.height / self.resize)))
      
        # number of objects in the image
        num_objs = len(annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        areas = []
        label = []
        for i in range(num_objs):

            xmin = annotation[i][0]
            ymin = annotation[i][1]
            xmax = xmin + annotation[i][2]
            ymax = ymin + annotation[i][3]
            l=annotation[i][4]
            area=annotation[i][2]*annotation[i][3]

            boxes.append([xmin/self.resize, ymin/self.resize, xmax/self.resize, ymax/self.resize])
          
            label.append(l)
            areas.append(area)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        labels = torch.as_tensor(label, dtype=torch.int64)
        img_id = torch.tensor([img_id])

        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation
        

    def __len__(self):
        return len(self.ids)

    def __len__(self):
        return len(self.filenames)

    
class ITBDataLoader:
    def __init__(self,pkl_path,img_path,batch,test_frac=0.2,valid_frac=0.2):
        self.path=pkl_path  #'D:/OBJ_PAPER/itb.pkl'
        self.img_path=img_path #'D:/OBJ_PAPER/Data/itb_p_500/'
        self.batch=batch
        self.test_frac=test_frac
        self.valid_frac=valid_frac 

    def collate_fn(self):
        return tuple(zip(*self.batch))

    def get_transform(self):
        custom_transforms = []
        custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)
    
    def train_valid_test_split(self):
        itb=pd.read_pickle(self.path)

        test=itb.sample(frac=self.test_frac,random_state=42)
        train_valid=itb.drop(test.index)
        valid=train_valid.sample(frac=self.valid_frac, random_state=42)
        train=train_valid.drop(valid.index)


        test.reset_index(drop=True,inplace=True)
        test['index']=test.index

        valid.reset_index(drop=True,inplace=True)
        valid['index']=valid.index

        train.reset_index(drop=True,inplace=True)
        train['index']=train.index
        print(' train shape : ',train.shape, 'valid shape : ', valid.shape, 'test shape : ', test.shape, )
           
        # create own Dataset
        train_dataset = simpleDataset(dataset=train,
                                    resize=4,
                                    color='L',
                                    img_foler_path=self.img_path,
                                    transforms=self.get_transform())

        valid_dataset = simpleDataset(
                                dataset=valid,
                                resize=4,
                                color='L',
                                img_foler_path=self.img_path,
                                transforms=self.get_transform())

        test_dataset = simpleDataset(
                                dataset=test,
                                resize=4,
                                color='L',
                                img_foler_path=self.img_path,
                                transforms=self.get_transform())



        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0,                                         
                                                collate_fn=self.collate_fn)

        valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=1,
                                                shuffle=True,
                                                num_workers=0,
                                                collate_fn=self.collate_fn)

        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=self.collate_fn)

        return test,valid,train,train_data_loader,valid_data_loader, test_data_loader
    
    def data_load(self,dataset):
        custom_dataset = simpleDataset(dataset=dataset,
                                    resize=4,
                                    color='L',
                                    img_foler_path=self.img_path,
                                    transforms=self.get_transform())
        
        data_loader = torch.utils.data.DataLoader(custom_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0,                                         
                                                collate_fn=self.collate_fn)
        return data_loader


class Valid_Infernce:
    def __init__(self,model,param,test_data_loader,confidence=[0.5]):
        self.model=model
        self.param=param
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.test_data_loader=test_data_loader
        self.confi=confidence
       

    def model_load(self):    
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.param))
        return self.model.eval()
    
            
    def valid(self,iou_threds,iteration,save=False,save_path=None):
        RECALL=[]
        PRECISION=[]
        F1_SCORE=[]
        MEAN_R=[]
        MEAN_P=[]
        MEAN_F1=[]
        model=self.model_load()

        for threshold in self.confi: 
            labels = []
            preds_adj_all = []
            annot_all = []
            print('confidence=',threshold)
            for im, annot in tqdm(self.test_data_loader, position = 0, leave = True):
                im = list(img.to(self.device) for img in im)
                for t in annot:
                    labels += t['labels']
                with torch.no_grad():
                    preds =model(im)
                    for id in range(len(preds)) :
                        idx_list = []

                        for idx, score in enumerate(preds[id]['scores']) :
                            
                            if score > threshold : 
                                idx_list.append(idx)

                        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
                        preds[id]['labels'] = preds[id]['labels'][idx_list]
                        preds[id]['scores'] = preds[id]['scores'][idx_list]
                    preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds]
                    preds_adj_all.append(preds_adj)
                    annot_all.append(annot)
        
            # iou_threds=0.5
            pr_f1=rp_f1.Precision_Recall_F1(iou_threds,preds_adj_all,annot_all)
        
            recall,mean_r=pr_f1.RECALL()
            precision,mean_p=pr_f1.PRECISION()
            f1_score,mean_f1=pr_f1.F1_SCORE()
        
            print(f'recall: {recall}, precision: {precision}, f1_score: {f1_score}')
            if save:
                RECALL.append(recall)
                PRECISION.append(precision)
                F1_SCORE.append(f1_score)
                MEAN_R.append(mean_r)
                MEAN_P.append(mean_p)
                MEAN_F1.append(mean_f1)
            
        print(f'mean_recall: {mean_r}, mean_precision: {mean_p}, mean_f1_score: {mean_f1}')

        if save:
            result=pd.DataFrame(self.confi,columns=['confidence'])
            result['iteration']=iteration
            result['recall']=[RECALL]
            result['mean_recall']=MEAN_R
            result['precision']=[PRECISION]
            result['mean_precision']=MEAN_P
            result['f1_form']=[F1_SCORE]
            result['mean_f1']=MEAN_F1
            result.to_csv(save_path,index=True)

        return 0
    
    def inference(self,img,threshold):
        model=self.model_load()
      
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