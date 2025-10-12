from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math
import numpy as np

class OR_Route:

    @staticmethod
    def create_data_model(dist_matrix_np,np_task,np_agv,agv_id,task_list):
        agv_info=np_agv[np_agv[:,-1]==agv_id]
        
        # 노드 0 = Depot
        distance_matrix = dist_matrix_np[np.ix_(task_list, task_list)]
        
        #time 설정
        task_time_deadline=np_task[np.isin(np_task[:, -1], task_list)][:,5]
        time_windows=[(0,time)for time in task_time_deadline]
        time_windows.insert(0,(0,0))

        # 각 노드 서비스 시간(초) — depot은 0
        service_times = list(np_task[np.isin(np_task[:, -1], task_list)][:,3])
        service_times = service_times.insert(0,0)

        return {
            "distance_matrix": distance_matrix,
            "time_windows": time_windows,
            "service_times": service_times,
            "num_vehicles": 1,
            "depot": 0,
            "speed_dps": agv_info[0][1],
            "max_route_time": 10**9,   # 차량 최대 운행시간(초)
            "max_route_distance": 10**9, # 차량 최대 거리(거리 단위)
        }

    @staticmethod
    def main(dist_matrix_np,np_task,np_agv,agv_id,task):
        np_task[:,-1]=np_task[:,-1]+1
        task_list=np.array(task)+1
        task_list=np.insert(task_list,0,0,axis=0)

        data = OR_Route.create_data_model(dist_matrix_np,np_task,np_agv,agv_id,task_list)
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]),
            data["num_vehicles"],
            data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)

        # ---- 거리 콜백 (총 거리 최소화 목적) ----
        def distance_callback(from_index, to_index):
            f = manager.IndexToNode(int(from_index))
            t = manager.IndexToNode(int(to_index))
            return int(round(data["distance_matrix"][f][t]))
        distance_cb_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_cb_idx)

        # ---- Distance Dimension (차량별 최대거리) ----
        routing.AddDimension(
            distance_cb_idx,
            0,  # slack
            int(round(data["max_route_distance"])),
            True,
            "Distance",
        )
        distance_dim = routing.GetDimensionOrDie("Distance")

        # ---- Time(이동시간+서비스시간) 콜백 ----
        def time_with_service_callback(from_index, to_index):
            f = manager.IndexToNode(int(from_index))
            t = manager.IndexToNode(int(to_index))
            dist = data["distance_matrix"][f][t]
            move_sec = dist / max(float(data["speed_dps"]), 1e-9)
            service_sec = int(data["service_times"][f])  # 출발 노드 서비스시간 포함
            return int(math.ceil(move_sec + service_sec))
        time_cb_idx = routing.RegisterTransitCallback(time_with_service_callback)

        # ---- Time Dimension (시간창 + 최대 운행시간) ----
        routing.AddDimension(
            time_cb_idx,
            0,  # waiting 허용: 창 내에서 자동으로 슬랙 사용
            int(data["max_route_time"]),
            True,  # start at zero
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        # ---- 시간창 적용 ----
        for node, (early, late) in enumerate(data["time_windows"]):
            index = manager.NodeToIndex(node)
            time_dim.CumulVar(index).SetRange(int(early), int(late))

        # 차량 시작/종료도 완화 목적함수에 포함(해 안정화)
        for v in range(data["num_vehicles"]):
            start = routing.Start(v)
            end = routing.End(v)
            # depot 창을 시작 노드에 적용
            depot_tw = data["time_windows"][data["depot"]]
            time_dim.CumulVar(start).SetRange(int(depot_tw[0]), int(depot_tw[1]))
            time_dim.CumulVar(end).SetRange(0, int(data["max_route_time"]))
            routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(start))
            routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(end))

        # ---- 탐색 파라미터 ----
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromSeconds(10)

        solution = routing.SolveWithParameters(params)
        if not solution:
            print("해를 찾지 못했습니다.")
            return

        # ---- 해 출력 ----
        total_dist = 0.0
        total_time = 0
        for v in range(data["num_vehicles"]):
            index = routing.Start(v)
            route_plan = []
            while not routing.IsEnd(index):
                t_cumul = solution.Value(time_dim.CumulVar(index))
                node = manager.IndexToNode(index)
                route_plan.append(task_list[node]-1)
                index = solution.Value(routing.NextVar(index))
            # End 노드
            t_cumul = solution.Value(time_dim.CumulVar(index))
            node = manager.IndexToNode(index)
            route_plan.append(task_list[node]-1)

            # 거리/시간 집계
            route_dist = solution.Value(distance_dim.CumulVar(index))
            route_time = t_cumul
            total_dist += route_dist
            total_time += route_time

        
        return route_plan[1:-1]

