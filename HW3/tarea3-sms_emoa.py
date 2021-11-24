from utils import get_extreme_points_c, save_json, mat_to_arr, extract_median_front
from pymoo.factory import get_problem, get_performance_indicator, get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from optproblems import Problem
from evoalgos.algo import SMSEMOA
from evoalgos.individual import ESIndividual
import math
import random
import numpy as np

RUNS = 20

problems = [{"problem":"wfg1", "k": 3, "pop_size":92, "n_gen":400}, 
{"problem":"wfg2", "k": 3, "pop_size":92, "n_gen":400},
{"problem":"wfg3", "k": 3, "pop_size":92, "n_gen":400},
{"problem":"wfg4", "k": 5, "pop_size":10, "n_gen":750},
{"problem":"wfg4", "k": 6, "pop_size":8, "n_gen":1000},
{"problem":"wfg4", "k": 7, "pop_size":6, "n_gen":1250},
{"problem":"wfg4", "k": 8, "pop_size":5, "n_gen":1500},
{"problem":"wfg4", "k": 9, "pop_size":4, "n_gen":1750},
{"problem":"wfg4", "k": 10, "pop_size":4, "n_gen":2000}]

# problems = [{"problem":"wfg1", "k": 3, "pop_size":92, "n_gen":400, "ref_points": get_reference_directions("das-dennis", 3, n_partitions=12)}, 
# {"problem":"wfg2", "k": 3, "pop_size":92, "n_gen":400, "ref_points": get_reference_directions("das-dennis", 3, n_partitions=12)},
# {"problem":"wfg3", "k": 3, "pop_size":92, "n_gen":400, "ref_points": get_reference_directions("das-dennis", 3, n_partitions=12)},
# {"problem":"wfg4", "k": 5, "pop_size":212, "n_gen":750, "ref_points": get_reference_directions("das-dennis", 5, n_partitions=6)},
# {"problem":"wfg4", "k": 6, "pop_size":195, "n_gen":1000, "ref_points": get_reference_directions("das-dennis", 6, n_partitions=4)},
# {"problem":"wfg4", "k": 7, "pop_size":175, "n_gen":1250, "ref_points": get_reference_directions("das-dennis", 7, n_partitions=3)},
# {"problem":"wfg4", "k": 8, "pop_size":156, "n_gen":1500, "ref_points": get_reference_directions("multi-layer",
#                                                                                                     get_reference_directions("das-dennis", 8, n_partitions=3, scaling=1.0),
#                                                                                                     get_reference_directions("das-dennis", 8, n_partitions=2, scaling=0.5),
#                                                                                                 )},
# {"problem":"wfg4", "k": 9, "pop_size":215, "n_gen":1750, "ref_points": get_reference_directions("multi-layer",
#                                                                                                     get_reference_directions("das-dennis", 9, n_partitions=3, scaling=1.0),
#                                                                                                     get_reference_directions("das-dennis", 9, n_partitions=2, scaling=0.5),
#                                                                                                 )},
# {"problem":"wfg4", "k": 10, "pop_size":275, "n_gen":2000, "ref_points": get_reference_directions("multi-layer",
#                                                                                                     get_reference_directions("das-dennis", 10, n_partitions=3, scaling=1.0),
#                                                                                                     get_reference_directions("das-dennis", 10, n_partitions=2, scaling=0.5),
#                                                                                                 )}]

class Analytics:
    def __init__(self, n_dim):
        super().__init__()
        self.front_history = {"fronts" : [], "hyper_volumes" : []}
        self.ideal_point = np.full(n_dim, np.inf)
        self.worst_point = np.full(n_dim, -np.inf)
        self.extreme_points = None
    
    def update(self, population, problem):
        pop = np.array([np.array(individual.phenome) for individual in population])
        F = problem.evaluate(pop)
        I = NonDominatedSorting().do(F, only_non_dominated_front=True)
        first_front = F[I, :]
        self.ideal_point = np.min(first_front, axis=0)
        self.worst_point = np.max(first_front, axis=0)
        epsilon = np.zeros(shape=self.worst_point.shape)
        epsilon.fill(1.)
        ref_point = self.worst_point + epsilon
        self.front_history["fronts"].append(first_front)
        hv = get_performance_indicator("hv", ref_point=ref_point)
        self.extreme_points = get_extreme_points_c(first_front, self.ideal_point)
        hypervolume = hv.do(np.vstack((self.extreme_points, self.ideal_point)))
        self.front_history["hyper_volumes"].append(hypervolume)

for run in range(1, RUNS + 1):
    analytics = {}
    for problem_spec in problems: 
        random.seed(run)
        problem_pymoo = get_problem(problem_spec["problem"], n_var=24, n_obj=problem_spec["k"]) 
        def obj_function(phenome):
            F = problem_pymoo.evaluate(np.array(phenome))
            return tuple([f_i for f_i in F])
        problem = Problem(obj_function, num_objectives=problem_spec["k"], max_evaluations=problem_spec["n_gen"], name="Example")
        population = []
        init_step_sizes = [0.25]
        for _ in range(problem_spec["pop_size"]):
            population.append(ESIndividual(genome=[500.*random.random() for _ in range(24)],
                                        learning_param1=1.0/math.sqrt(problem_spec["k"]),
                                        learning_param2=0.0,
                                        strategy_params=init_step_sizes,
                                        recombination_type="none",
                                        num_parents=1))
        analytics_obj = Analytics(problem_spec["k"])
        ea = SMSEMOA(problem, population, problem_spec["pop_size"], num_offspring=1)   
        stop = False
        while not stop:
            try:
                unevaluated = []
                for individual in population:
                    if individual.date_of_birth is None:
                        individual.date_of_birth = ea.generation
                    individual.date_of_death = None
                    if not individual.objective_values:
                        unevaluated.append(individual)
                ea.problem.batch_evaluate(unevaluated)
                ea.step()
                analytics_obj.update(population, problem_pymoo)              
                stop = ea.stopping_criterion()
            except StopIteration as e:
                stop = True
        analytics_obj.front_history["fronts"] = [mat_to_arr(extract_median_front(analytics_obj.front_history["fronts"], analytics_obj.front_history["hyper_volumes"])), mat_to_arr(analytics_obj.front_history["fronts"][-1])]
        analytics["SMS-EMOA-" + problem_spec["problem"] + "-k=" + str(problem_spec["k"])] = analytics_obj.front_history
        print("Finished optimizing: " + "SMS-EMOA-" + problem_spec["problem"] + "-k=" + str(problem_spec["k"]))  
    save_json('analytics-sms-emoa', analytics)