from re import DEBUG, S
from hklearn_genetic.utils import ProblemUtils
from hklearn_genetic.Evaluators.FunctionEvaluators import DTLZ1, DTLZ2, DTLZ12D
from hklearn_genetic.Problems import RealGAProblem, BaseGAProblem, RealBGAProblem, BinaryGAProblem
from hklearn_genetic.NSGAIII import NSGAIII
from hklearn_genetic.NSGAII import NSGAII
import json
import os
import os.path
import copy

ANALITYCS = True
BINARY = False
BGA = True
SBX = False
DEBUG = False
RUNS = 5

def save_json(path, analytics):
    with open(path + ".json", 'a') as fp:           
        fp.write('\n')
        json.dump(analytics, fp)

parameter_set = [[],[],[]]

analytics = {#"DTLZ1-NSGA2-Binary" : {"evaluations" : [], "phenotypes":[]},
    #"DTLZ1-NSGA3-Binary" : {"evaluations" : [], "phenotypes":[]},  
    #"DTLZ2-NSGA2-Binary" : {"evaluations" : [], "phenotypes":[]},  
    #"DTLZ2-NSGA3-Binary" : {"evaluations" : [], "phenotypes":[]}, 
    #"DTLZ1-NSGA2-Real-BGA" : {"evaluations" : [], "phenotypes":[]},  
    "DTLZ1-NSGA3-Real-BGA" : {"evaluations" : [], "phenotypes":[]},  
    #"DTLZ2-NSGA2-Real-BGA" : {"evaluations" : [], "phenotypes":[]},  
    #"DTLZ2-NSGA3-Real-BGA" : {"evaluations" : [], "phenotypes":[]}, 
    #"DTLZ1-NSGA2-Real-SBX" : {"evaluations" : [], "phenotypes":[]},  
    #"DTLZ1-NSGA3-Real-SBX" : {"evaluations" : [], "phenotypes":[]},  
    #"DTLZ2-NSGA2-Real-SBX" : {"evaluations" : [], "phenotypes":[]},  
    #"DTLZ2-NSGA3-Real-SBX" : {"evaluations" : [], "phenotypes":[]}
    }
dtlz1 = DTLZ1()
dtlz2 = DTLZ2()
problems = {"DTLZ1-NSGA2-Binary" : {"algorithm": NSGAII(tournament_type = 1),"parameters" : {"n_individuals":200,"evaluator": dtlz1, "thresh": -0.001, "bounds" : (0, 1), "max_iter":1000, "pc":.9, "pm":1/324, "n_prec":8, "n_dim":12}}, 
    "DTLZ1-NSGA3-Binary" : {"algorithm":NSGAIII(),"parameters" : {"n_individuals": 100, "evaluator": dtlz1, "thresh": -0.001, "bounds" : (0, 1), "max_iter":1000, "pc":.9, "pm":1/324, "n_prec":8, "n_dim":12}}, 
    "DTLZ2-NSGA2-Binary" : {"algorithm":NSGAII(tournament_type = 1),"parameters" : {"n_individuals":100, "evaluator": dtlz2, "thresh": -0.001, "bounds" : (0, 1), "max_iter":400, "pc":.9, "pm":1/324, "n_prec":8, "n_dim":12}}, 
    "DTLZ2-NSGA3-Binary" : {"algorithm":NSGAIII(),"parameters" : {"n_individuals":100, "evaluator": dtlz2, "thresh": -0.001, "bounds" : (0, 1), "max_iter":400, "pc":.9, "pm":1/324, "n_prec":8, "n_dim":12}},
    # "DTLZ1-NSGA2-Real-BGA" : {"algorithm":NSGAII(tournament_type = 1),"parameters" : {"n_individuals":200, "evaluator": dtlz1, "thresh": -0.001, "bounds" : (0, 1), "max_iter":1000, "pc":.9, "pm":1/12, "n_dim":12}}, 
    "DTLZ1-NSGA3-Real-BGA" : {"algorithm":NSGAIII(),"parameters" : {"n_individuals":92, "evaluator": dtlz1, "thresh": -0.001, "bounds" : (0, 1), "max_iter":400, "pc":.9, "pm":1/12, "n_dim":12}}, 
    # "DTLZ2-NSGA2-Real-BGA" : {"algorithm":NSGAII(tournament_type = 1),"parameters" : {"n_individuals":100, "evaluator": dtlz2, "thresh": -0.001, "bounds" : (0, 1), "max_iter":400, "pc":1., "pm":1/12, "n_dim":12}}, 
    # "DTLZ2-NSGA3-Real-BGA" : {"algorithm":NSGAIII(),"parameters" : {"n_individuals":92, "evaluator": dtlz2, "thresh": -0.001, "bounds" : (0, 1), "max_iter":400, "pc":1., "pm":1/12, "n_dim":12}},
    #"DTLZ1-NSGA2-Real-SBX" : {"algorithm":NSGAII(tournament_type = 1),"parameters" : {"n_individuals":200, "evaluator": dtlz1, "thresh": -0.001, "bounds" : (0, 1), "max_iter":1000, "pc":.9, "pm":1/12, "n_dim":12}}, 
    "DTLZ1-NSGA3-Real-SBX" : {"algorithm":NSGAIII(),"parameters" : {"n_individuals":92, "evaluator": dtlz1, "thresh": -0.001, "bounds" : (0, 1), "max_iter":1000, "pc":1., "pm":1/12, "n_dim":12}}, 
    #"DTLZ2-NSGA2-Real-SBX" : {"algorithm":NSGAII(tournament_type = 1),"parameters" : {"n_individuals":100, "evaluator": dtlz2, "thresh": -0.001, "bounds" : (0, 1), "max_iter":400, "pc":.9, "pm":1/12, "n_dim":12}}, 
    #"DTLZ2-NSGA3-Real-SBX" : {"algorithm":NSGAIII(),"parameters" : {"n_individuals":100, "evaluator": dtlz2, "thresh": -0.001, "bounds" : (0, 1), "max_iter":400, "pc":.9, "pm":1/12, "n_dim":12}} 
    }




def StartRuns(number_of_runs, problem_specifications):
    for run in range(number_of_runs):
        for k, v in problem_specifications.items(): 
            print(f"Starting {k} Optimization: {run + 1}")   
            if "Binary" in k and BINARY:
                problem = BinaryGAProblem._BaseBinaryGAProblem(v["parameters"])
            elif "Binary" in k and not BINARY:
                continue
            if "BGA" in k and BGA:
                problem = RealBGAProblem._BaseRealBGAProblem(v["parameters"])
            elif "BGA" in k and not BGA:
                continue
            if "SBX" in k and SBX:
                problem = RealGAProblem._BaseRealGAProblem(v["parameters"])
            elif "SBX" in k and not SBX:
                continue
            v["algorithm"].max_iter = v["parameters"]["max_iter"]
            final_front = v["algorithm"].evolve(problem, v["parameters"]["n_individuals"], debug = DEBUG)
        
            fitness = [list(p.fitness_metric) for p in final_front]
            phenotypes = [list(p.phenotype) for p in final_front]
            analytics[k]["evaluations"] = fitness
            analytics[k]["phenotypes"] = phenotypes
            params = copy.deepcopy(v["parameters"])
            params.pop("evaluator")
            analytics[k]["parameters"] = params
        # Dump analytics in a json in order to extract insights from it offline
        save_json('analytics', analytics)

StartRuns(RUNS, problems)
