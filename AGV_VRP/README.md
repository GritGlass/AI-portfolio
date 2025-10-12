# 🚢 Smart Port AGV Route Optimization (Smart Port AGV VRP)

**Repository**: [https://github.com/GritGlass/AI-portfolio/tree/master/AGV_VRP](https://github.com/GritGlass/AI-portfolio/tree/master/AGV_VRP)

> Project based on the *Smart Maritime Logistics AI Mission Challenge – Smart Port AGV Route Optimization Competition*.
> AGVs start from the depot (0,0), visit each task once, and return to the depot while jointly optimizing **schedule + route** under constraints (capacity, max_distance) and deadlines.

---

## 📷 Example

**AGV Routing Demo**
Multiple AGVs operate on a Manhattan grid-based port environment and construct **round-trip routes** (DEPOT → tasks → DEPOT). Below is a visualization example (sample data):

* Left: Depot and task coordinates with route overlay (visit sequence)
* Right: Task completion time, lateness status, cumulative travel and service time per AGV table

> Actual visual examples are generated in `/examples` and `/output`. (See **Run** section below)

---

## 🔧 How it Works (Pipeline)

1. **Input Load**

   * Load task data from `tasks.csv`,or JSON, including task coordinates ((x, y)), `service_time`, `deadline`, and `demand`.
   * Load AGV specifications from `agv.csv`, including `speed_cells_per_sec`, `capacity`, and `max_distance`.

2. **Initialization**

   * Generate an initial solution using Nearest Neighbor (NN/Greedy) or Clarke–Wright Savings.
   * Apply automatic split routing (round-trip split) and task reassignment when constraints are violated.

3. **Local Search**

   * Apply local search operators such as 2-opt, Or-opt, Swap, Relocate, and Exchange.
   * Perform both inter-route (between routes) and intra-route (within a route) optimization.

4. **Global Search**

   * Apply metaheuristics such as NSGA-II, Artificial-Bee-Colony(ABC).
   * Configure Google OR-Tools VRP to model Time Windows/Capacity as hard constraints.

5. **Scheduling & Time Accumulation**

   * Travel time: $\text{time}=\frac{\text{distance}}{\text{speed\cells\per\sec}}$
   * Task service: accumulate `service_time` upon arrival at each task.
   * Completion time: accumulated time up to that task + `service_time`
   * Lateness: completion time > `deadline`

6. **Objective Calculation**

   * Total score = total travel time + total service time + total lateness penalty
    ```math
    t_\text{move} = \frac{\text{Manhattan distance}}{\text{speed\_cells\_per\_sec}}
    ```
---

## ⚙️ Rules & Constraints

* **Movement Rules**

  * All AGVs start simultaneously and **collisions are assumed to be avoided**.
  * Grid-based sensor path → **Manhattan distance** is used.
  * Travel time is **accumulated** and **does not reset** when returning to the depot.
  * ( \text{time} = \frac{\text{distance}}{\text{speed_cells_per_sec}} )

* **Task Rules**

  * Upon arrival, each task is processed immediately for its `service_time`.
  * Completion time = accumulated travel/service time + `service_time`.
  * If the `deadline` is exceeded ->  **지각** is applied.
  * Tasks **cannot be split**

* **Round-Trip Constraints (DEPOT → tasks → DEPOT)**

  * `capacity` **must not be exceeded.**
  * `max_distance` **must not be exceeded.**
  * Constraints **reset after returning** to the depot.
  * **Empty return trips are not allowed** (must visit at least one task before returning).

---

## 📦 Installation

```bash
# 1) Clone
git clone https://github.com/GritGlass/AI-portfolio.git
cd AI-portfolio/AGV_VRP

# 2) (Optional) Create venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

---


## 🧪 Tech Stack

| Tool/Lib                       | Purpose                              |
| ------------------------------ | -------------------------------------|
| **NumPy / Pandas**             | Data processing, score/time accumulation  |
| **NetworkX / Matplotlib**      | Graph construction, route and Gantt chart visualization |
| **Google OR-Tools**            | Modeling VRP with Time Windows/Capacity |

---

## 🙏 Acknowledgements

* This project is inspired by the Smart Maritime Logistics AI Mission Challenge – Smart Port AGV Route Optimization. Key problem definitions and rules from the competition are summarized in this README.

---

## 📄 License

This repository is provided for research and educational purposes only. For any commercial use, please contact the author beforehand.

---

## 📬 Contact

* Author: **GritGlass (YU H.)**
* Repo: [https://github.com/GritGlass/AI-portfolio/tree/master/AGV_VRP](https://github.com/GritGlass/AI-portfolio/tree/master/AGV_VRP)
