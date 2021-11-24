"""
Implementation of the NSGAIII MOEA
Author: Luis Andrés Eguiarte-Morett (Github: @leguiart)
License: MIT. 
"""
from numpy.core.shape_base import block
from .utils import PlotUtils, ProblemUtils
from numpy.core.fromnumeric import shape
from hklearn_genetic.Problems.IProblem import IProblem
from pymoo.factory import get_reference_directions
from pymop.factory import get_problem
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy

PLAY = True


class NSGAIII:
    def __init__(self, max_iter : int = 5000):
        self.max_iter = max_iter
        self.structured_ref_points = list(get_reference_directions("das-dennis", 3, n_partitions=12))
        self.tournament_type = 1
        self.k_tournament = 2

    def evolve(self, problem : IProblem, n_individuals : int, debug : bool = False):
        #Initialize population
        its = 0       
        self.ideal_point = None
        self.worst_point = None
        self.nadir_point = np.array([1.,1.,1.])
        self.extreme_points = []
        self.epsilon_nad = 1e-6
        self.debug = debug

        parent_pop = problem.populate(n_individuals)

        if debug:
            dtlz1_pf = get_problem("dtlz1", n_var=12, n_obj=3).pareto_front(np.array(self.structured_ref_points))
        while its < self.max_iter:     
            child_pop = problem.generate(parent_pop)
            population = copy.deepcopy(parent_pop) + child_pop
            problem.evaluate(population)
            fronts = self._non_dominated_sorting(population)
            # if debug:
            #     front_mats = []
            #     for front in fronts:
            #         front_mat = ProblemUtils._to_matrix_fitness(front)
            #         front_mats += [front_mat]
            #     plot_mat_2d(front_mats, "Fronts", "")
            parent_pop = []
            last_front = []
            s_t = []
            i = 0
            while len(s_t) < n_individuals:
                s_t += fronts[i]
                i += 1
            last_front = fronts[i - 1]
            if len(s_t) == n_individuals:
                parent_pop = s_t
                continue
            else:
                for j in range(i - 1):
                    parent_pop += fronts[j]

                k = n_individuals - len(parent_pop)
                # print('Parent pop')
                if debug:
                    s_t_mat = ProblemUtils._to_matrix_fitness(s_t)
                    PlotUtils.plot_mat_3d_lines([dtlz1_pf, np.array(self.structured_ref_points), s_t_mat], "Pre-normalization", f"Gen {its + 1}", play = PLAY)

                self._normalize(s_t, population, fronts[0])
                if debug:
                    # parent_pop_mat = ProblemUtils._to_matrix_fitness(parent_pop)
                    first_front_mat = ProblemUtils._to_matrix_fitness(fronts[0])
                    PlotUtils.plot_mat_3d_lines([dtlz1_pf, np.array(self.structured_ref_points), first_front_mat], "Post-normalization", f"", play = PLAY)

                #Associate each member of parent_pop in the normalized space, with a reference point
                self._associate(s_t, self.structured_ref_points)

                #Compute niche count for each reference point
                niche_counts = [0 for i in range(len(self.structured_ref_points))]
                for p in parent_pop:
                    niche_counts[p.reference_line[1]]+=1        
                self._niching(k, niche_counts, self.structured_ref_points, last_front, parent_pop)

                if debug:
                    parent_pop_mat = ProblemUtils._to_matrix_fitness(parent_pop)
                    PlotUtils.plot_mat_3d_lines([dtlz1_pf, np.array(self.structured_ref_points), parent_pop_mat], "Post-Niching", f"", play = PLAY)
            #parent_pop = self._select(problem, parent_pop)
            its+=1

        problem.evaluate(parent_pop)
        return self._non_dominated_sorting(parent_pop)[0]

    def _normalize(self, s_t, r_t, first_front):
        obj_num = len(s_t[0].fitness_metric)

        #Translating objectives
        #Calculate the ideal point with the best values observed so far in the optimization run
        self.ideal_point = _calculate_ideal(self.ideal_point, r_t)
        #Calculate the worst point with the worst values observed so far in the optimization run
        self.worst_point  = _calculate_worst(self.worst_point, r_t)

        #Computing extreme points
        s_t_eval= np.array([p.fitness_metric for p in first_front])
        self.extreme_points = _calculate_extreme_points(obj_num, s_t_eval, self.extreme_points, self.ideal_point)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        self.nadir_point = _calculate_nadir_point(self.extreme_points, self.epsilon_nad, obj_num, self.ideal_point, self.worst_point, first_front, r_t)

        if self.debug:
            s_t_mat = ProblemUtils._to_matrix_fitness(s_t)
            PlotUtils.plot_mat_2d([np.array(self.structured_ref_points), s_t_mat, self.extreme_points, self.ideal_point.reshape(1, len(self.ideal_point)), self.worst_point.reshape(1, len(self.worst_point)), self.nadir_point.reshape(1, len(self.nadir_point))], 
            "Ideal and extreme points", f"", labels=["ref points", "population members", "extreme points", "ideal point", "worst point", "nadir point"], play = PLAY)

        for s in s_t:
            denom = self.nadir_point - self.ideal_point
            #Prevent numerical errors
            denom[denom == 0] = 1e-12
            s.fitness_metric = (s.fitness_metric - self.ideal_point)/denom
                       
    def _associate(self, s_t, ref_points):
        for s in s_t:
            min_val = float('inf')
            w_min = None
            line_index = 0
            for i, w in enumerate(ref_points):
                d = _perpendicular_dist(s.fitness_metric, w)
                if min_val > d:
                    min_val = d
                    w_min = w
                    line_index = i
            s.reference_line = (w_min, line_index)
            s.distance = (min_val, line_index)

    def _niching(self, K, niche_counts, ref_points, last_front, parent_pop):
        k = 0
        ref_points_copy = ref_points.copy()
        while k < K:
            #Get the niches with minimum associated points
            min_val = min(niche_counts)
            j_min = []
            for i, count in enumerate(niche_counts):
                if count == min_val:
                    j_min += [i]
            #Choose a niche from the ones with less associated points at random
            j = random.choice(j_min)
            #Get the elements in the last front associated with the niche j
            I_j = []
            for s in last_front:
                if (s.reference_line[0] == ref_points_copy[j]).all():
                    I_j += [copy.deepcopy(s)]
            
            #If there are elements in the front associated with the niche j
            if len(I_j) != 0:
                s_to_add = None
                #If there are no elements from the previous generation parent population associated with niche j, but there are elements in the last front associated with niche j
                if niche_counts[j] == 0:
                    #get the element in the last front which is associated to the reference line from the niche j such that it has a minimal distance to the reference line
                    min_val = float('inf')
                    s_min = None
                    for s in I_j:
                        if min_val > s.distance[0]:
                            min_val = s.distance[0]
                            s_min = s
                    #We will add this element to the parent population of the next generation
                    s_to_add = s_min
                #If there are no elements from the previous generation parent population associated with niche j and there are elements in the last front associated with niche j 
                else:
                    #Just choose one element at random
                    s_to_add = random.choice(I_j)
                parent_pop += [s_to_add]
                niche_counts[j] += 1
                index = 0
                for i in range(len(last_front)):
                    if last_front[i] is s_to_add:
                        index = i
                        break
                last_front.pop(index)
                k += 1
            #If there are no elements in the front associated with the niche j
            else:
                #Stop considering that niche for this generation
                niche_counts.pop(j)
                ref_points_copy.pop(j)

    def _select(self, problem : IProblem, population : list):
        """Evaluates the population of solutions based on the problem and 
        applies the chosen selection operator over the evaluated proposed solutions
        Parameters
        ----------
        problem : IProblem
            problem object wich implements the IProblem interface
        population : list
            list of objects, each representing a proposed solution
        """
        population = problem.evaluate(population)
        # population.sort(key = lambda x : x.fitness_metric)
        #fitness_metrics = [individual.fitness_metric for individual in population]
        child_population = []
        t = 0
        #with replacement
        if self.tournament_type == 1:
            
            while t < len(population):
                tournament_contestants = np.random.permutation(len(population))[0:self.k_tournament]
                greatest_score_so_far = float('inf')
                tournament_winner = None
                for contestant in tournament_contestants:
                    if population[contestant].rank < greatest_score_so_far:
                        greatest_score_so_far = population[contestant].rank
                        #population[t] = copy.deepcopy(population[contestant])
                        tournament_winner = copy.deepcopy(population[contestant])
                child_population += [tournament_winner]
                t+=1
        #without replacement
        elif self.tournament_type == 0:
            while t < len(population):
                permutation = np.random.permutation(len(population))
                i = 0
                tournament_winner = None
                while i < len(permutation) and t < len(population):
                    greatest_score_so_far = float('inf')
                    for j in range(i,min(i + self.k_tournament, len(population))):
                        if population[permutation[j]].rank < greatest_score_so_far:
                            greatest_score_so_far = population[j].rank
                            tournament_winner = copy.deepcopy(population[permutation[j]])
                            #population[t] = 
                    child_population += [tournament_winner]
                    t+=1
                    i+=self.k_tournament
        return child_population

    def _non_dominated_sorting(self, population : list):
        fronts = [[]]
        for i in range(len(population)):
            population[i].dominated = []
            population[i].domination_counter = 0

            for j in range(len(population)):
                if i != j:
                    if population[i].dominates(population[j]):
                        population[i].dominated.append(population[j])
                    elif population[j].dominates(population[i]):
                        population[i].domination_counter += 1
            if population[i].domination_counter == 0:
                fronts[0] += [population[i]]
                population[i].rank = 1
        #Initialize front counter
        i = 0
        while len(fronts[i]) != 0:
            next_front_members = []
            for p in fronts[i]:
                for q in p.dominated:
                    q.domination_counter -= 1
                    if q.domination_counter == 0:
                        q.rank = i + 1
                        next_front_members += [q]
            fronts.append(next_front_members)
            i+=1
                
        fronts.pop()
        return fronts


def _calculate_extreme_points(obj_num, evaluations, previous_extreme_points, ideal_point):
    # calculate the asf which is used for the extreme point decomposition
    weights = np.eye(obj_num)
    weights[weights == 0] = 1e6

    # add the old extreme points to never loose them for normalization
    _F = evaluations
    if len(previous_extreme_points) != 0:
        _F = np.concatenate([previous_extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * weights[:, None, :], axis=2)

    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points
    # epsilon = 1e6
    # w = np.array([epsilon for i in range(obj_num)])
    # def set_to_zero(x):
    #     x[x <= 1e-3] = 0.
    #     return x
    # evals = list(map(set_to_zero, evaluations))
    # A = evals + previous_extreme_points
    # e = []
    # for i in range(obj_num):
    #     if i - 1 > -1:
    #         w[i - 1] = epsilon
    #     w[i] = 1
    #     e += [min_asf(A, w, ideal_point)]
    # return e

def _calculate_nadir_point(extreme_points, epsilon_nad, dimensions, ideal_point, worst_point, first_front, r_t):
    #Obtaining intercepts
    #We have to handle certain special cases      
    b = False
    nadir_point = []
    mat_extreme_points = extreme_points - ideal_point
    worst_of_front = np.array(_get_maximals_in_dim(first_front))
    worst_of_population = np.array(_get_maximals_in_dim(r_t))
    #Non-invertible matrix
    if np.linalg.det(mat_extreme_points) == 0:
        b = True
    else:
        coefficients = np.linalg.inv(mat_extreme_points)@np.ones(shape = dimensions)
        intercepts = 1./coefficients
        nadir_point = intercepts + ideal_point
        #Check for negative intercepts (or smaller than an epsilon_nad threshold)
        #and that the resulting nadir point estimation is not larger than the worst point
        if (intercepts < epsilon_nad).any() or not np.allclose(np.dot(mat_extreme_points, coefficients), np.ones(shape = dimensions)) or (nadir_point > worst_point).any():
            b = True
        else:
            boo = nadir_point > worst_point
            nadir_point[boo] = worst_point[boo]

    #Fall back to the maximum in each objective for the current front
    if b:
        nadir_point = np.array(worst_of_front)

    
    b = nadir_point - ideal_point <= epsilon_nad
    nadir_point[b] = worst_of_population[b]
    return nadir_point

#The distance from a line with director vector w which contains point z to a point s 
def _perpendicular_dist(s, w):
    t = np.dot(w, s) / np.dot(w,w)
    v_d = s - t*w
    return np.sqrt(np.dot(v_d, v_d))

def min_asf(solutions, w, v):
    #We get the point from A with a minimal ASF function value
    min_val = float('inf')
    candidate_extreme_point = None
    for p in solutions:
        asf = _ASF(p, w, v)
        if min_val > asf:
            min_val = asf
            candidate_extreme_point = p
    return candidate_extreme_point

def _ASF(f, w, ideal):
    li = [(f[i] - ideal[i])/w[i] for i in range(len(f))]
    return max(li)

def _calculate_ideal(previous_ideal, solutions):
    M = len(solutions[0].fitness_metric)
    #Calculate the ideal point
    if previous_ideal is None:
        previous_ideal = _get_minimals_in_dim(solutions)
    else:
        _update_minimals_in_dim(solutions, previous_ideal)
                
    return previous_ideal

def _calculate_worst(previous_worst, solutions):
    M = len(solutions[0].fitness_metric)
    #Calculate the ideal point
    if previous_worst is None:
        previous_worst = _get_maximals_in_dim(solutions)
    else:
        _update_maximals_in_dim(solutions, previous_worst)
    return previous_worst

def _get_minimals_in_dim(solutions):
    dimensions = len(solutions[0].fitness_metric)
    minimals = np.zeros(shape=dimensions)
    for i in range(dimensions):
        min_val = float('inf')
        for p in solutions:
            if min_val > p.fitness_metric[i]:
                min_val = p.fitness_metric[i]
        minimals[i] = min_val
    return minimals

def _get_maximals_in_dim(solutions):
    dimensions = len(solutions[0].fitness_metric)
    maximals = np.zeros(shape=dimensions)
    for i in range(dimensions):
        max_val = float('-inf')
        for p in solutions:
            if max_val < p.fitness_metric[i]:
                max_val = p.fitness_metric[i]
        maximals[i] = max_val
    return maximals

def _update_minimals_in_dim(solutions, previous_minimals):
    for i in range(len(previous_minimals)):
        for p in solutions:
            if previous_minimals[i] > p.fitness_metric[i]:
                previous_minimals[i] = p.fitness_metric[i]

def _update_maximals_in_dim(solutions, previous_maximals):
    for i in range(len(previous_maximals)):
        for p in solutions:
            if previous_maximals[i] < p.fitness_metric[i]:
                previous_maximals[i] = p.fitness_metric[i]

