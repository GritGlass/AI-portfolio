import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import glob
import wandb as wb

from mealpy.swarm_based import PSO
from mealpy.utils.space import CategoricalVar
from mealpy import ABC, HHO, ZOA ,AEO, WaOA, TDO, ServalOA
import re, logging

class WandbHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.PAT = re.compile(
            r"Epoch:\s*(\d+),\s*Current best:\s*([-\d\.eE]+),\s*Global best:\s*([-\d\.eE]+),\s*Runtime:\s*([-\d\.eE]+)"
        )

    def emit(self, record):
        msg = record.getMessage()
        m = self.PAT.search(msg)   
        if m:
            epoch = int(m.group(1))
            cur_best = float(m.group(2))
            glob_best = float(m.group(3))
            runtime = float(m.group(4))
            wb.log({
                "epoch": epoch,
                "current_best": cur_best,
                "global_best": glob_best,
                "runtime": runtime
            })

class OBP:
    def __init__(self,MODEL_NAME,DIST_MATRIX,SAMPLE,LOCATION, K,ALPHA,DEPOT=0):
        '''  
        K : job 개수 (cluster)
        ALPHA : fitness 계산식 가중치
        DEPOT : 이동거리 계산시 원점 지정, 0번째 slot에서 출발 후 0번째 slot으로 도착하는 거리 계산
        DIST_MATRIX : 거리 행렬
        SAMPLE : 주문 번호, 상품 slot 번호 정보가 있는 주문 데이터
        '''
        #=========dataset===============
        self.model_name=MODEL_NAME
        self.dist_matrix=DIST_MATRIX
        self.sample=SAMPLE[['order_num','slot']]
        self.location=LOCATION[['aisle','rack','bay','slot','x','y']]

        #=========hyper paramer==========
        self.k = K 
        self.n = len(self.sample.order_num.unique())
        self.a = ALPHA
        self.depot = DEPOT
       
    def _get_dist(self, a, b):
        ''' 두 slot의 거리 계산 '''
        return float(self.dist_matrix[a, b])
    
    def _time_consumption(self,job_dist):
        ''' 시간=거리/속력, 성인 평균 걸음 속도= 80m/m (미터,분) '''
        max_distance=job_dist.max()
        return max_distance/80
    
    def _prior_items(self,batch_order):
        items=[]
        
        batch_order_prior=batch_order.merge(self.location,how='left',on='slot')
        batch_order_prior.sort_values(by=['aisle','bay','rack'],inplace=True)

        slot_list=batch_order_prior['slot'].to_list()
        start=slot_list.pop(0)
        items.append(start)

        while slot_list:
            slot_dist={rest_slot: self.dist_matrix[start,rest_slot] for rest_slot in slot_list}
            next_slot=sorted(slot_dist.items(), key=lambda item: item[1])[0][0]
            items.append(next_slot)
            start=next_slot
            slot_list.remove(next_slot)
            
        return items

    def _cal_dist(self,batch_order)->float:

        ''' batch 별로 거리 계산 '''
        
        items=self._prior_items(batch_order)
        
        n = len(items)
        if n == 0:
            return 0.0
        if n == 1:
            # depot -> item -> depot
            return self._get_dist(self.depot, items[0]) + self._get_dist(items[0], self.depot)

        total = 0.0
        # 인접 이동
        for i in range(n - 1):
            total += self._get_dist(items[i], items[i+1])
        # depot 왕복
        total += self._get_dist(self.depot, items[0])
        total += self._get_dist(items[-1], self.depot)
        return total

    def order_distance(self,job)->np.array:
        '''job 별 거리 값 array 반환 [dist1,dist2,dist3,dist4]'''
        # job: 길이 N 의 정수(클러스터 ID) 배열
        orders = self.sample.order_num.unique()
        groups = {c: orders[job == c] for c in np.unique(job)}  # {c1: [order ids], c2: [order ids]}
        batch_dist = []
        for batch_id, order_num in groups.items():
            batch_order = self.sample[self.sample['order_num'].isin(order_num)]
            total_distance = self._cal_dist(batch_order)
            batch_dist.append(total_distance)
        return np.array(batch_dist, dtype=float)

    
    def fitness(self,solution):
        ''' 최종 목적 함수 계산  '''
        sol = np.asarray(solution)

        #sloution이 카테고리가 아니라 numeric float로 생성될 경우-> int로 변경
        # if np.issubdtype(sol.dtype, np.floating):
        #     job = np.rint(sol).astype(int) % self.k
        # else:
        job = sol.astype(int)

        job_dist = self.order_distance(job)
        if job_dist.size == 0:
            return 0.0
        time_cost=self._time_consumption(job_dist)

        obj = self.a *time_cost + (1-self.a)*job_dist.var()
        return float(obj)
    
    def fitness2(self,solution):
        """
        job_dist: 길이 K 배열 (각 job의 총 이동 거리, meters)
        handle_times: 길이 K 배열 (각 job의 피킹/처리 시간, minutes). 없으면 0.
        """
        speed_m_per_min=80.0
        handle_times=None,
        a=0.8 
        alpha=5.0
        lam_bal=1.0
        lam_avg=1e-3
        eps=1e-9
        
        sol = np.asarray(solution)
        job = sol.astype(int)

        job_dist = self.order_distance(job)

        job_dist = np.asarray(job_dist, dtype=float)
        K = len(job_dist)
        if handle_times is None:
            handle_times = np.zeros(K, dtype=float)
        else:
            handle_times = np.asarray(handle_times, dtype=float)

        # 각 배치 시간 (분)
        T = job_dist / speed_m_per_min + handle_times

        # 스무딩된 max (LSE)
        lse = (1.0/alpha) * np.log(np.exp(alpha*T).sum())

        # 균형: 변동계수(CV) -> 스케일 불변
        mean_T = T.mean()
        std_T  = T.std(ddof=0)
        cv_T   = std_T / (mean_T + eps)

        # 전체 평균(미세 타이브레이커)
        avg_T = mean_T

        J = a * lse + (1 - a) * lam_bal * cv_T + lam_avg * avg_T
        return float(J)


    def model(self,params):
        ''' params: {epoch: , popsize: , nlimits: , seed: }  '''
        if self.model_name=='ABC':
            return ABC.OriginalABC(epoch=params.get('epoch'), pop_size=params.get('popsize'), n_limits = params.get('nlimits'), seed=params.get('seed'))
        elif self.model_name=='AEO':
            return AEO.AugmentedAEO(epoch=params.get('epoch'), pop_size=params.get('popsize'),seed=params.get('seed'))
        elif self.model_name=='WaOA':
            return WaOA.OriginalWaOA(epoch=params.get('epoch'), pop_size=params.get('popsize'),seed=params.get('seed'))
        elif self.model_name=='ServalOA':
            return ServalOA.OriginalServalOA(epoch=params.get('epoch'), pop_size=params.get('popsize'),seed=params.get('seed'))
        elif self.model_name=='STO':
            return TDO.OriginalTDO(epoch=params.get('epoch'), pop_size=params.get('popsize'),seed=params.get('seed'))
        elif self.model_name=='HHO':
            return HHO.OriginalHHO(epoch=params.get('epoch'), pop_size=params.get('popsize'))
        elif self.model_name=='SPO':
            return PSO.OriginalPSO(epoch=params.get('epoch'),c1=2.0,c2=2.0,pop_size=params.get('popsize'))



    def run(self,model):
        bounds = [CategoricalVar(valid_sets=tuple(range(self.k))) for _ in range(self.n)] # [1],[1],[2],[2],[0],[3],... : order id
       
        problem = {
            "obj_func": self.fitness2,   
            "bounds": bounds,
            "minmax": "min",
        }
        # model =HHO.OriginalHHO(epoch=1000, pop_size=1000)
        # model = self.model(params)
        best = model.solve(problem)

        print("최적 해:", best.solution)
        print("최적 값:", best.target)
        return best
    
    def result(self,best,wh_location,save_path):
        sample_orders=self.sample.order_num.unique()
        result=pd.DataFrame({'order_num':sample_orders, 'job':best.solution.tolist()})
        sample2=self.sample.copy()
        final=sample2.merge(result,how='left',on='order_num')[['slot','job']]
        final_location=wh_location.merge(final,how='left',on='slot')
        final_location.fillna('-1',inplace=True)
        plt.figure(figsize=(20,10))
        sns.scatterplot(data=final_location,x='x',y='y',hue='job')
        plt.title(f'fitness: {best.target}')
        plt.savefig(save_path,dpi=300)
        plt.show()
        plt.close()
        return print(f'Save plot {save_path} done!')

def get_sample(dataset)->pd.DataFrame:
    total_order_num=len(dataset.order_num.unique())
    #파일 중 일부 order만 샘플링 후 수행
    sample_order_num=np.random.random_integers(1,total_order_num,1) #N: 몇개의 order를 뽑을 것인지
    sample_order=np.random.choice(total_order_num,size=sample_order_num,replace=False) #선택된 order_id
    sample1=dataset[dataset['order_num'].isin(sample_order)] 
    print(f'주문 id: {sample_order}')
    return sample1, sample_order_num, sample_order

