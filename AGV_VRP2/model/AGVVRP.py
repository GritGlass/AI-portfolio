import pandas as pd
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import random
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from deap import base, creator, tools
from typing import List, Iterable, Optional, Set, Tuple
from collections import Counter
import sys
import json
import itertools
from .OR_routing import OR_Route
import logging 
import datetime
from collections import defaultdict



now=datetime.datetime.now()
log_name=now.strftime("%y%m%d-%H-%M-%S")
log_path = os.path.join(os.path.dirname(__file__), f"../log/{log_name}.log")
log_path = os.path.join(os.path.dirname(__file__), f"../log/{log_name}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler()                           # 콘솔 출력
    ]
)


class AGVVRPOPTIM:
    def __init__(self,how,agv:np.array,task:np.array,dist_matrix:np.array,population:int, global_search_iteration:int, local_search_iter:int,init_params:dict,save_path=None):
        ''' population은 반드시 짝수로 할 것 '''
        self.agv_np=agv
        self.task_np=task
        self.dist_matrix=dist_matrix
        
        self.population=population

        self.agv_num=len(self.agv_np[:,-1])
        self.task_num=len(self.task_np[:,-1])

        self.init_params=init_params

        self.global_search_iteration=global_search_iteration
        self.local_search_iteration=local_search_iter

        self.save_path=save_path

        self.how=how #ga, random

    def _time_dist_clustering_init(self,)-> dict:
        '''
        [agv_id, agv_id, ...]
        '''
        rand_state=random.randint(0,4294967295)
        X=self.task_np[:,1:3]
        X=np.column_stack((X,self.dist_matrix[-1][1:]))
        kmeans = KMeans(n_clusters=self.agv_num, random_state=rand_state, n_init="auto").fit(X)
        rand_jobs=[job for job in kmeans.labels_]
        assert len(rand_jobs) == self.task_num
        return rand_jobs
    
    def _dist_clustering_init(self,)-> dict:
        '''
        [agv_id, agv_id, ...]
        '''
        rand_state=random.randint(0,4294967295)
        X=self.task_np[:,1:3]
        kmeans = KMeans(n_clusters=self.agv_num, random_state=rand_state, n_init="auto").fit(X)
        rand_jobs=[job for job in kmeans.labels_]
        assert len(rand_jobs) == self.task_num
        return rand_jobs
    
    def _random_init(self,) -> dict:
        '''
        [agv_id, agv_id, ...]
        '''
        rng = np.random.default_rng(seed=10**9)

        # 기본 최소치 배분
        base = np.full(self.agv_num, 1, dtype=int)
        remain = self.task_num - self.agv_num * 1
        if remain < 0:
            raise ValueError("1 * self.agv_num > self.task_num 이라서 조건 불만족")

        # 남은 값을 다항분포로 랜덤 분배
        extra = rng.multinomial(remain, np.ones(self.agv_num) / self.agv_num)
        counts = base + extra

        rand_jobs=[[agv]*task_num for agv,task_num in zip(list(range(self.agv_num)),counts)]
        rand_jobs=list(itertools.chain.from_iterable(rand_jobs))
        random.shuffle(rand_jobs)

        assert len(rand_jobs) == self.task_num

        return list(rand_jobs)
    
    def _del_duplicate(self,solutions:np.array):
        '''
        solutions : [[solution1],[solution2],[solution3],...]

        '''
        unique_solution=list(set(tuple(sole) for sole in solutions))
        
        return [list(sol) for sol in unique_solution]
    
    def _check_duplicate(self,solutions:np.array):
        '''
        input 
            solutions : [[sol1],[sol2]]
        output 
            result    : False(합격)/True(불합격)
        '''
        def check(solutions):
            check_list=defaultdict(list)
            for idx,sol in enumerate(solutions):
                    check_list[tuple(sol)].append(idx)
            return [index for sol,index in check_list.items() if len(index)>1]

        const4_list=check(solutions)
        const4=(len(const4_list)>0)
        if const4==False:
            result=False
        else:
            result=True
        return result,const4_list
    
    def _chromosome_check(self,solutions:np.array,type=None):
        '''
        if any of agv has no task, delete the solutions
        type 
            - None                   :  [[sol1],[sol2],[sol3]]
            - sol_key_fitness_dict   :  {(sol):fiteness, (sol2):fitness,(sol3):fitness}
            - id_key_solfit_dict     :  {0:[[sol1],fiteness],1:[[sol2],fiteness],2:[[sol3],fiteness]}
        return : input format
        '''
        if type=='sol_key_fitness_dict':
            checked_solutions={init_sol:fit for init_sol,fit in list(solutions.items()) if self.agv_num==len(np.unique(init_sol))}
            return checked_solutions
        elif type=='id_key_solfit_dict':
            checked_solutions={id:[init_sol,fit] for id,(init_sol,fit) in list(solutions.items()) if self.agv_num==len(np.unique(init_sol))}
            return checked_solutions
        elif type=='one':
            if self.agv_num==len(np.unique(solutions)):
                checked_solutions=solutions
            else:
                checked_solutions=[]
            return checked_solutions
        elif type=='list':
            checked_solutions=[init_sol for init_sol in solutions if self.agv_num==len(np.unique(init_sol))]
            return checked_solutions
        else:
            
            const4,_=self._check_duplicate(solutions)
            const2_list=[0 if self.agv_num==len(np.unique(sol)) else 1 for sol in solutions]
            const2=(1 in const2_list)
            if (const2==False)&(const4==False):
                    #합격
                    result=False
            else:
                    #불합격
                    result=True
            return result
        
    
    def initalization(self)->np.array:
        '''     
        _time_dist_clustering_init : generate initial solutions by time_dist_clustering
        _dist_clustering_init : generate initial solutions by dist_clustering
        _random_init : generate initial solutions by random

        return : [[solution1],[solution2],[solution3],...]
        '''
        initial_funcions=[self._time_dist_clustering_init]*int(self.population*self.init_params.get('time_dist'))
        initial_funcions+=[self._dist_clustering_init]*int(self.population*self.init_params.get('dist'))
        initial_funcions+=[self._random_init]*int(self.population*self.init_params.get('random'))

        inital_solutions=[init_sol() for init_sol in initial_funcions]

        #delete duplicate 
        inital_solutions=self._del_duplicate(inital_solutions)
        #chromosome chekcing 
        inital_solutions_re=self._chromosome_check(inital_solutions)

        #population number check : Add the lacking number of solutions randomly
        while inital_solutions_re:
            inital_solutions.append(self._random_init())
            inital_solutions=self._del_duplicate(inital_solutions)
            inital_solutions=self._chromosome_check(inital_solutions)
            if len(inital_solutions)==self.population:
                inital_solutions_re=False
        assert len(inital_solutions)==self.population
        return inital_solutions
    

    def pox(self,solutions_fitness,parents):
        """
        solutions_fitness   : solution 과 fitness 정보 리스트 
        parents             : selection op로 선택된 부모 인덱스 리스트 [p1_idx,p2_idx]
        k                   : change할 인덱스 개수
        반환값               : child1, child2
        """
        p1_idx,p2_idx=parents
        p1=list(solutions_fitness[p1_idx][0])
        p2=list(solutions_fitness[p2_idx][0])

        chrom_check=True
      
        # while chrom_check:

        c1=p2.copy()
        c2=p1.copy()

        genome=np.random.choice(list(range(self.agv_num)),size=random.randint(0,14),replace=False)
        p1_genome_idx=np.where(~np.isin(p1,genome))[0]
        p2_genome_for_c1 = [x for x in p2 if x not in genome]

        p2_genome_idx=np.where(~np.isin(p2,genome))[0]
        p1_genome_for_c2 = [x for x in p1 if x not in genome]

        for pid,g in zip(p1_genome_idx,p2_genome_for_c1):
            c1[pid]=g

        for pid,g in zip(p2_genome_idx,p1_genome_for_c2):
            c2[pid]=g

        chrom_check=self._chromosome_check([c1,c2])
        if chrom_check:
            c1=self._random_init()
            c2=self._random_init()
            

        return [c1,c2]
    
    def jbx(self,solutions_fitness,parents):
        """
        solutions_fitness   : solution 과 fitness 정보 리스트 
        parents             : selection op로 선택된 부모 인덱스 리스트 [p1_idx,p2_idx]
        k1                  : change할 인덱스 개수
        k2                  : p2에서 change할 인덱스 개수
        반환값               : child1, child2
        """
        p1_idx,p2_idx=parents
        p1=list(solutions_fitness[p1_idx][0])
        p2=list(solutions_fitness[p2_idx][0])

    
        c2=p2.copy()
        c1=p1.copy()
        genome1=list(np.random.choice(list(range(self.agv_num)),size=random.randint(0,14),replace=False))
        genome2=list(set(list(range(self.agv_num)))-set(genome1))

        p1_genome_idx=np.where(~np.isin(p1,genome1))[0]
        p2_genome_for_c1 = [x for x in p2 if x not in genome1]

        for pid,g in zip(p1_genome_idx,p2_genome_for_c1):
            c1[pid]=g

        p2_genome_idx=np.where(~np.isin(p2,genome2))[0]
        p1_genome_for_c2 = [x for x in p1 if x not in genome2]
        for pid,g in zip(p2_genome_idx,p1_genome_for_c2):
            c2[pid]=g

        chrom_check=self._chromosome_check([c1,c2])
        if chrom_check:
            c1=self._random_init()
            c2=self._random_init()


        return [c1,c2]
    
    def _child_mutation(self,solutions_fitness,parents:list)->list:
        p1_idx,p2_idx=parents
        
        p1=list(solutions_fitness[p1_idx][0])
        p2=list(solutions_fitness[p2_idx][0])
        p1_result=True
        while p1_result:
            random.shuffle(p1)
            p1_result=self._chromosome_check([p1])
            
        
        p2_result=True
        while p2_result:
            random.shuffle(p2)
            p2_result=self._chromosome_check([p2])
    
    
        return [p1, p2]
    
    def _parent_elite(self,solutions_fitness)->list:
        solutions_fitness=sorted(solutions_fitness.items(), key=lambda x: x[1][1])
        p1_idx=solutions_fitness[0][0]
        p2_idx=solutions_fitness[1][0]
        return [p1_idx,p2_idx]
    
    def _parent_roulette_wheel(slef,solutions_fitness)->list:
        parent_idx=list(solutions_fitness.keys())
        p=np.array([fitness for sol,fitness in list(solutions_fitness.values())])
        prob = p / p.sum()
        return np.random.choice(parent_idx,size=2,p=prob,replace=False).tolist()
        
    def _parent_random(self,solutions_fitness)->list:
        return np.random.choice(list(solutions_fitness.keys()),size=2).tolist()
    
    def _parent_operator(self,):
        parent_rn=np.random.random()
        if parent_rn<=0.4:
            return self._parent_elite
        elif 0.4<parent_rn<=0.8:
            return self._parent_roulette_wheel
        else:
            return self._parent_random

    def _child_parent_pair(self):
        #child operator 
        child_crossover_pox=[self.pox]*int((self.population/2)*self.init_params.get('pox'))
        child_crossober_jbx=[self.jbx]*int((self.population/2)*self.init_params.get('jbx'))
        child_mutation=[self._child_mutation]*int((self.population/2)*self.init_params.get('mutation'))
        
        #parent operator
        child_parent=[(child,self._parent_operator()) for child in child_crossover_pox]
        child_parent+=[(child,self._parent_operator()) for child in child_crossober_jbx]
        child_parent+=[(child,self._parent_operator()) for child in child_mutation]
        return child_parent
    
    def _local_search(self,solutions:list,)->dict:
        prior_solutions=self.search_task_prior(solutions)
        solutions_fitness={idx: [list(solutions[idx]),self.fitness(sol)] for idx,sol in enumerate(prior_solutions)}
        
        for a in range(self.local_search_iteration):
            if self.how=='ga':
                #1.child_op와 parent_op pair 생성
                child_parent_op=self._child_parent_pair()
                
                #2.chile_op와 해당 parent index 리스트 생성
                child_op_parent_id_pair=[[values[0],values[1](solutions_fitness)] for idx,values in enumerate(child_parent_op)]
                
                #3.child_op로 자식 생성 -> new solutions
                #  개별 solution 체크 : chromosome 체크 -> 모든 agv에 task가 할당 되었는지 -> child_op안에 추가
                #  new_solutions 내에서 중복 체크 -> 중복인 solution이 있다면 삭제 후 동일한 child_op로 다시 생성
                child_solutions_pair=[child_op(solutions_fitness,parent_idx) for child_op, parent_idx in child_op_parent_id_pair]
                child_solutions=list(itertools.chain(*child_solutions_pair)) #[sol1,sol2,sol3,...]
                
                #부모-자식 id, len(2/population)=len(parent_id)
                parent_id=[op[1] for op in child_op_parent_id_pair]
                parent_id=[[pid_p]*2 for pid_p in parent_id]
                parent_id=list(itertools.chain(*parent_id))

                child_id_pair=[[idx*2,idx*2+1]for idx in range(len(child_solutions_pair))]
                child_id=list(itertools.chain(*child_id_pair))
                
                child_parent_id={cid:parent_id[idx] for idx,cid in enumerate(child_id)}
                #child_solutions에서 중복이 있는지 check    
                child_check_result,child_duplicate_index=self._check_duplicate(child_solutions)
                
                if child_check_result: #중복있음,True
                    child_solutions=self._del_duplicate(child_solutions)
                    # child_solutions=list(np.where(~np.isin(child_solutions,child_duplicate_index))) #중복 제거
                            
                    for idx,del_idx_list in enumerate(child_duplicate_index):
                        for del_idx in del_idx_list[1:]:
                            child_op,parent=child_op_parent_id_pair[int(del_idx//2)]
                            
                            app=True
                            for i in range(3):
                                re_child1,re_child2=child_op(solutions_fitness,parent)

                                if re_child1 not in child_solutions:
                                    child_solutions.insert(del_idx,re_child1)
                                    app=False
                                    break

                                elif re_child2 not in child_solutions:
                                    child_solutions.insert(del_idx,re_child2)
                                    app=False
                                    break

                            if app:
                                while app:
                                    rand_sol=self._random_init()
                                    if rand_sol not in child_solutions:
                                        child_solutions.insert(del_idx,re_child2)
                                        app=False

                #child_solution_fitness 계산
                prior_child_solutions=self.search_task_prior(child_solutions)
                child_solutions_fitness={idx: [list(child_solutions[idx]),self.fitness(sol)] for idx,sol in enumerate(prior_child_solutions)}
                #4. 2번의 parent index로 생성된 부모-자식의 fitneess 비교
                # 부모 index - 자식 index 매핑
                for cid,ps_id in child_parent_id.items():
                    child_sol,child_fit=child_solutions_fitness[cid]
                    for p in ps_id:
                        parent_sol,parent_fit=solutions_fitness[p]
                        if child_fit < parent_fit:
                            solutions_fitness.update({p:[child_sol,child_fit]})

            elif self.how=='random':
                child_solutions=[self._random_init() for _ in range(self.population)]
                #child_solutions에서 중복이 있는지 check    
                child_check_result,child_duplicate_index=self._check_duplicate(child_solutions)
                
                if child_check_result: #중복있음,True
                    child_solutions=self._del_duplicate(child_solutions)
                    app=True
                    if app:
                        while app:
                            rand_sol=self._random_init()
                            if rand_sol not in child_solutions:
                                child_solutions.insert(del_idx,re_child2)
                                app=False

                #기존 solution과 새로운 세대 병합
                total_solution=child_solutions+solutions
                total_solution=self._del_duplicate(total_solution)
                prior_total_solution=self.search_task_prior(total_solution) #[{agv_id:[task]},{agv_id:[task]}]
                solutions_fitness={idx:[total_solution[idx],self.fitness(sol)] for idx,sol in enumerate(prior_total_solution)}
                solutions_fitness={idx:[sol,fiteness] for idx,(sol,fiteness) in sorted(solutions_fitness.items(), key=lambda x: x[1][1])}
                sol_list=list(solutions_fitness)[:self.population]
                solutions_fitness = {k: solutions_fitness[k] for k in sol_list if k in solutions_fitness}

            assert len(child_solutions)==self.population

            
            # ------------------------ best solution log -------------------------------------------------------------
            sort_solutions_fitness=list(sorted(solutions_fitness.items(), key=lambda x: x[1][1]))
            local_solution=sort_solutions_fitness[0][1][0]
            local_fitness=sort_solutions_fitness[0][1][1]

            local_solution_dict={x: np.where(np.array(local_solution) == x)[0].tolist() for x in range(self.agv_num)}
            logging.info(f'local iteration : {a}, solution : {local_solution_dict} -> fitness : {local_fitness}')

        return solutions_fitness

    
    def search_job_dispatching(self,solutions:list):
        '''
        solutions : [(solution1), (solution2), (solution3), (solution4),...]
        '''
        # best_fitness=float('inf')
        
        for g in range(self.global_search_iteration):
            prior_solutions=self.search_task_prior(solutions) #[{agv_id:[task]},{agv_id:[task]}]
            solutions_fitness={tuple(solutions[idx]):self.fitness(sol) for idx,sol in enumerate(prior_solutions)}
            
            # ------------------------ best solution log -------------------------------------------------------------
            sort_solutions_fitness=list(sorted(solutions_fitness.items(), key=lambda x: x[1]))
            global_solution=sort_solutions_fitness[0][0]
            global_fitness=sort_solutions_fitness[0][1]

            global_solution_dict={x: np.where(np.array(global_solution) == x)[0].tolist() for x in range(self.agv_num)}
            logging.info(f'global iteration : {g}, solution : {global_solution_dict} -> fitness : {global_fitness}')
            
            # -------------------------------save the best solution -------------------------------------------------------------

            if self.save_path:
                os.makedirs(os.path.join(os.path.dirname(__file__), f"../result/solution/"),exist_ok=True)
                            
                with open(os.path.join(os.path.dirname(__file__), f"../result/solution/{global_fitness}_fitness.json"), "w", encoding="utf-8") as f:
                    json.dump(global_solution_dict, f, ensure_ascii=False, indent=4)

            # --------------------------------------------------------------------------------------------------------
            update_solutions_fitness=self._local_search(solutions,)
            
            update_solutions=[sol for sol, fit in update_solutions_fitness.values()]
            
            #기존 solution과 새로운 세대 병합
            total_solution=update_solutions+solutions
            total_solution=self._del_duplicate(total_solution)
            prior_total_solution=self.search_task_prior(total_solution) #[{agv_id:[task]},{agv_id:[task]}]
            total_solutions_fitness={tuple(total_solution[idx]):self.fitness(sol) for idx,sol in enumerate(prior_total_solution)}
            updated_solutions=[sol for sol, fiteness in sorted(total_solutions_fitness.items(), key=lambda x: x[1])]
            

            #하위 20% 까지 삭제
            drop_rate=self.init_params.get('global_drop_rate')
     
            if len(updated_solutions)>=int((self.population*(1-drop_rate))):
                #하위 20% 까지 삭제
                updated_solutions=updated_solutions[:int((self.population*(1-drop_rate)))]
              
            #random으로 새로운 해 추가
         
            while len(updated_solutions)!=self.population:
                new_solution=self._random_init()
                new_solution=self._chromosome_check(new_solution,type='one')
                if (len(new_solution)>0)&(new_solution not in updated_solutions):
                    updated_solutions.append(list(new_solution))
            
            assert len(updated_solutions)==self.population, 'the number of solution is wrong'
            solutions=updated_solutions
            
        return solutions

    def search_task_prior(self,solutions:np.array)-> list:
        '''
        solutions : [(solutions1),(solutions2),(solutions3),...]
        result : [{agv1:[task1,task2,task3],...}] 
        
        '''
        
        def sort_one_solution(solution):
             return {x: np.where(np.array(solution) == x)[0].tolist() for x in range(self.agv_num)}
        
        result = [sort_one_solution(sol) for sol in solutions] #[{sol1},{sol2},{sol3}]

        for i,sol in enumerate(result):
            for agv_id,task_list in sol.items():
                
                route_plan=OR_Route.main(self.dist_matrix,self.task_np,self.agv_np,agv_id,task_list)
                result[i][agv_id]=route_plan

        return result

    def _go_to_depot(self,agv_id,current_position,current_time,total_distance):
        max_distance=self.agv_np[self.agv_np[:,-1]==agv_id][0][2]
        max_capacity=self.agv_np[self.agv_np[:,-1]==agv_id][0][3]
        total_distance+=self.dist_matrix[current_position+1][0]
        current_time+=self.dist_matrix[current_position+1][0]/self.agv_np[self.agv_np[:,-1]==agv_id][0][1]
        current_position=0
        return max_distance,max_capacity,current_position,current_time,total_distance

    
    def fitness(self,solution:dict)->float:
        ''' 
        input : one solution,{agv_id:[task1,task2,task3,...],...},agv별 task는 prior 기준으로 sorting 되어 있음
        output : fitness(float)

        '''
        fitness_distance=0
        fitness_time=0

        for agv_id in range(self.agv_num):
            task_list=solution[agv_id] 

            max_distance=self.agv_np[self.agv_np[:,-1]==agv_id][0][2]
            max_capacity=self.agv_np[self.agv_np[:,-1]==agv_id][0][3]
            current_time=0
            time_panelty=0
            total_distance=0
            current_position=0

            for tid in task_list:
                
                moved_distance=self.dist_matrix[0][tid+1]
                 #task로 이동 불가능 -> depot -> task 이동 
                if ((max_distance-moved_distance)<=0) or ((max_distance-moved_distance-self.dist_matrix[tid+1][0])<=0):
                    #current_position=depot, capacity, max 초기화, currnet,total +
                    max_distance,max_capacity,current_position,current_time,total_distance=self._go_to_depot(agv_id,current_position,current_time,total_distance)
                                                                                                               

                    #task로 다시 이동
                    total_distance+=moved_distance
                    max_distance-=moved_distance
                    current_time+=(moved_distance/self.agv_np[self.agv_np[:,-1]==agv_id][0][1])
                    current_position=tid

                    #task 수행
                    max_capacity-=self.task_np[tid][4]
                    current_time+=self.task_np[tid][3]
                    

                #task로 이동 가능
                else:
                    total_distance+=moved_distance
                    max_distance-=moved_distance
                    current_time+=(moved_distance/self.agv_np[self.agv_np[:,-1]==agv_id][0][1])
                    current_position=tid

                    #capacity 가능
                    if (max_capacity-self.task_np[tid][4])<0:
                        max_distance,max_capacity,current_position,current_time,total_distance=self._go_to_depot(agv_id,current_position,current_time,total_distance)
                                                                                                     
                        #task로 다시 이동
                        total_distance+=moved_distance
                        max_distance-=moved_distance
                        current_time+=(moved_distance/self.agv_np[self.agv_np[:,-1]==agv_id][0][1])
                        current_position=tid

                        #task 수행
                        max_capacity-=self.task_np[tid][4]
                        current_time+=self.task_np[tid][3]
                    

                    else:
                        #task 수행
                        max_capacity-=self.task_np[tid][4]
                        current_time+=self.task_np[tid][3]
                        current_position=tid
                        

                if self.task_np[tid][5]<current_time:
                    time_panelty+=(current_time-self.task_np[tid][5])*10 

                fitness_distance+=total_distance
                fitness_time+=time_panelty

        return float(fitness_distance+fitness_time) #object function
    
    
    def run(self,):
        init_solution=self.initalization()
        print(f'start: population - {len(init_solution)}')
        generate_solutions=self.search_job_dispatching(init_solution)
        prior_solutions=self.search_task_prior(generate_solutions)

        solutions_fitness={tuple(generate_solutions[idx]):self.fitness(sol) for idx,sol in enumerate(prior_solutions)}
            
        if self.save_path:
            sort_solutions_fitness=list(sorted(solutions_fitness.items(), key=lambda x: x[1]))
            global_solution=sort_solutions_fitness[0][0]
            global_fitness=sort_solutions_fitness[0][1]
      
            global_solution_dict={x: np.where(np.array(global_solution) == x)[0].tolist() for x in range(self.agv_num)}
            with open(os.path.join(self.save_path,f"{global_fitness}_fitness.json"), "w", encoding="utf-8") as f:
                json.dump(global_solution_dict, f, ensure_ascii=False, indent=4)

        print(f'{global_solution} : {global_fitness}')     
        return global_solution
    
    def random_run(self,):
        init_solution=self.initalization()
        print(f'start: population - {len(init_solution)}')
        generate_solutions=self.search_job_dispatching(init_solution)
        prior_solutions=self.search_task_prior(generate_solutions)

        solutions_fitness={tuple(generate_solutions[idx]):self.fitness(sol) for idx,sol in enumerate(prior_solutions)}
            
        if self.save_path:
            sort_solutions_fitness=list(sorted(solutions_fitness.items(), key=lambda x: x[1]))
            global_solution=sort_solutions_fitness[0][0]
            global_fitness=sort_solutions_fitness[0][1]
      
            global_solution_dict={x: np.where(np.array(global_solution) == x)[0].tolist() for x in range(self.agv_num)}
            with open(os.path.join(self.save_path,f"{global_fitness}_fitness.json"), "w", encoding="utf-8") as f:
                json.dump(global_solution_dict, f, ensure_ascii=False, indent=4)

        print(f'{global_solution} : {global_fitness}')     
        return global_solution

# if __name__ == "__main__":
#     dist_matrix=pd.read_pickle('E:/glass_git/AGV_VRP/data/distance_matrix.pkl').to_numpy()
#     task=pd.read_csv('E:/glass_git/AGV_VRP/data/task_re.csv').to_numpy()
#     agv=pd.read_csv('E:/glass_git/AGV_VRP/data/agv_re.csv').to_numpy()

#     save_path='E:/glass_git/AGV_VRP/log/'
#     # save_path=E:/glass_git/AGV_VRP/result/solution
#     population=100
#     global_search_iteration=10
#     local_search_iteration=2
#     init_params={'time_dist':0.4,'dist':0.4,'random':0.2, 'pox':0.4, 'jbx':0.4, 'mutation':0.2, 'global_drop_rate':0.2}

#     agvvrp=AGVVRPOPTIM(agv,task,dist_matrix,population,global_search_iteration,local_search_iteration,init_params,save_path)
#     best_solution=agvvrp.run()
#     print(best_solution)
