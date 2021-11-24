from hklearn_genetic.utils import ProblemUtils
from hklearn_genetic.Evaluators.FunctionEvaluators import BaseBeale, BaseEggholder, BaseHimmelblau, BaseRastrigin, Sphere, Schaffer
from hklearn_genetic.Problems import RealBGAProblem, RealGAProblem, IntegerGAProblem, BinaryGAProblem
from hklearn_genetic.EvolutionaryAlgorithm import EvolutionaryAlgorithm
import numpy as np
import json
import os
import os.path

class MyAnalytics:
    def __init__(self, function_name):
        self.analytics = {"name" : function_name,"eval_mat" : [], "best_fitness" : [], "average_fitness":[]}

    def gather_analytics(self, X_eval):
        X_eval_li, X_eval_mat = ProblemUtils._to_evaluated_matrix(X_eval)
        self.analytics["eval_mat"]+=[X_eval_li]
        self.analytics["best_fitness"]+=[list(X_eval_mat[int(np.argmin(np.array(X_eval_mat[:, 2]), axis = 0)), :])]
        self.analytics["average_fitness"]+=[float(X_eval_mat[:, 2].mean())]


analytics_sphere = MyAnalytics("sphere")
analytics_schaffer = MyAnalytics("schaffer")
#Defining evaluators
sphere = Sphere()
schaffer = Schaffer()

#Defining problems
#Real SBX crossover and polynomial mutation
sphere_real_sbx = RealGAProblem._BaseRealGAProblem(sphere, -0.0001, (-5., 5.), pc = 0.85, pm = 0.1, analytics = analytics_sphere)
schaffer_real_sbx = RealGAProblem._BaseRealGAProblem(schaffer, -0.001, (-500, 500), pc = 0.85, pm = 0.1, analytics = analytics_schaffer)

#Defining the evolutionary algorithm
ea = EvolutionaryAlgorithm(selection="tournament", tournament_type=1)



print(ea.evolve(sphere_real_sbx, 100))
print(ea.evolve(schaffer_real_sbx, 100))

# Dump analytics in a json in order to extract insights from it offline
if os.path.isfile('sphere_analytics.json'):
    with open('sphere_analytics.json', 'a') as fp:           
        fp.write('\n')
        json.dump(analytics_sphere.analytics, fp)
else:
    with open('sphere_analytics.json', 'w') as fp: 
        json.dump(analytics_sphere.analytics, fp)

if os.path.isfile('schaffer_analytics.json'):
    with open('schaffer_analytics.json', 'a') as fp:           
        fp.write('\n')
        json.dump(analytics_schaffer.analytics, fp)
else:
    with open('schaffer_analytics.json', 'w') as fp: 
        json.dump(analytics_schaffer.analytics, fp)