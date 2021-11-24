
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from momap_elites import MOMAP_Elites
from utils import problems, get_problem, Analytics, matr_add_diff_size, mat_to_arr, extract_median_front, save_json
from pymoo.optimize import minimize
from pymoo.factory import get_problem, get_reference_directions

RUNS = 20

problems = [{"problem":"wfg1", "k": 3, "pop_size":92, "n_gen":400, "ref_points": get_reference_directions("das-dennis", 3, n_partitions=12)}, 
{"problem":"wfg2", "k": 3, "pop_size":92, "n_gen":400, "ref_points": get_reference_directions("das-dennis", 3, n_partitions=12)},
{"problem":"wfg3", "k": 3, "pop_size":92, "n_gen":400, "ref_points": get_reference_directions("das-dennis", 3, n_partitions=12)},
{"problem":"wfg4", "k": 5, "pop_size":212, "n_gen":750, "ref_points": get_reference_directions("das-dennis", 5, n_partitions=6)},
{"problem":"wfg4", "k": 6, "pop_size":195, "n_gen":1000, "ref_points": get_reference_directions("das-dennis", 6, n_partitions=4)},
{"problem":"wfg4", "k": 7, "pop_size":175, "n_gen":1250, "ref_points": get_reference_directions("das-dennis", 7, n_partitions=3)},
{"problem":"wfg4", "k": 8, "pop_size":156, "n_gen":1500, "ref_points": get_reference_directions("multi-layer",
                                                                                                    get_reference_directions("das-dennis", 8, n_partitions=3, scaling=1.0),
                                                                                                    get_reference_directions("das-dennis", 8, n_partitions=2, scaling=0.5),
                                                                                                )},
{"problem":"wfg4", "k": 9, "pop_size":215, "n_gen":1750, "ref_points": get_reference_directions("multi-layer",
                                                                                                    get_reference_directions("das-dennis", 9, n_partitions=3, scaling=1.0),
                                                                                                    get_reference_directions("das-dennis", 9, n_partitions=2, scaling=0.5),
                                                                                                )},
{"problem":"wfg4", "k": 10, "pop_size":275, "n_gen":2000, "ref_points": get_reference_directions("multi-layer",
                                                                                                    get_reference_directions("das-dennis", 10, n_partitions=3, scaling=1.0),
                                                                                                    get_reference_directions("das-dennis", 10, n_partitions=2, scaling=0.5),
                                                                                                )}]

def start_runs():
    for run in range(1, RUNS + 1):
        analytics = {}
        for problem_spec in problems:      
            problem = get_problem(problem_spec["problem"], n_var=24, n_obj=problem_spec["k"])
            # create the algorithm object
            algorithm = MOMAP_Elites(pop_size=problem_spec["pop_size"],
                            ref_dirs=problem_spec["ref_points"])
            call_back = Analytics(problem_spec["k"])
            # execute the optimization
            res = minimize(problem,
                        algorithm,
                        seed=run,
                        termination=('n_gen', problem_spec["n_gen"]),
                        callback = call_back)
            call_back.front_history["fronts"] = [mat_to_arr(extract_median_front(call_back.front_history["fronts"], call_back.front_history["hyper_volumes"])), mat_to_arr(call_back.front_history["fronts"][-1])]
            analytics["MOMAP_Elites-" + problem_spec["problem"] + "-k=" + str(problem_spec["k"])] = call_back.front_history
            print("Finished optimizing: " + "MOMAP_Elites-" + problem_spec["problem"] + "-k=" + str(problem_spec["k"]))  


            problem = get_problem(problem_spec["problem"], n_var=24, n_obj=problem_spec["k"])
            # create the algorithm object
            algorithm = NSGA3(pop_size=problem_spec["pop_size"],
                            ref_dirs=problem_spec["ref_points"])
            call_back = Analytics(problem_spec["k"])
            # execute the optimization
            res = minimize(problem,
                        algorithm,
                        seed=run,
                        termination=('n_gen', problem_spec["n_gen"]),
                        callback = call_back)
            call_back.front_history["fronts"] = [mat_to_arr(extract_median_front(call_back.front_history["fronts"], call_back.front_history["hyper_volumes"])), mat_to_arr(call_back.front_history["fronts"][-1])]
            analytics["NSGA3-" + problem_spec["problem"] + "-k=" + str(problem_spec["k"])] = call_back.front_history
            print("Finished optimizing: " + "NSGA3-" + problem_spec["problem"] + "-k=" + str(problem_spec["k"]))  

            algorithm = NSGA2(pop_size=problem_spec["pop_size"])
            call_back = Analytics(problem_spec["k"])
            res = minimize(problem,
                        algorithm,
                        ('n_gen', problem_spec["n_gen"]),
                        seed=run,
                        verbose=False,
                        callback = call_back)
            call_back.front_history["fronts"] = [mat_to_arr(extract_median_front(call_back.front_history["fronts"], call_back.front_history["hyper_volumes"])), mat_to_arr(call_back.front_history["fronts"][-1])]
            analytics["NSGA2-" + problem_spec["problem"] + "-k=" + str(problem_spec["k"])] = call_back.front_history
            print("Finished optimizing: " + "NSGA2-" + problem_spec["problem"] + "-k=" + str(problem_spec["k"])) 

            algorithm = MOEAD(problem_spec["ref_points"],  n_neighbors=problem_spec["pop_size"])
            call_back = Analytics(problem_spec["k"])
            res = minimize(problem,
                        algorithm,
                        ('n_gen', problem_spec["n_gen"]),
                        seed=run,
                        verbose=False,
                        callback = call_back)
            call_back.front_history["fronts"] = [mat_to_arr(extract_median_front(call_back.front_history["fronts"], call_back.front_history["hyper_volumes"])), mat_to_arr(call_back.front_history["fronts"][-1])]
            analytics["MOEAD-" + problem_spec["problem"] + "-k=" + str(problem_spec["k"])] = call_back.front_history
            print("Finished optimizing: " + "MOEAD-" + problem_spec["problem"] + "-k=" + str(problem_spec["k"])) 
        save_json('analytics', analytics)

start_runs()