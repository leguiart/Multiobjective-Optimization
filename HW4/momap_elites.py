

import numpy as np
import random
from pymoo.core.survival import Survival
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import intersect
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga3 import NSGA3, HyperplaneNormalization, associate_to_niches, comp_by_cv_then_random

class MOMAP_Elites(NSGA3):
    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                 crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=MultiObjectiveDisplay()):
        super().__init__(ref_dirs, pop_size=pop_size, sampling=sampling, selection=selection, crossover=crossover, mutation=mutation, eliminate_duplicates=eliminate_duplicates, n_offsprings=n_offsprings, display=display, survival = ReferenceDirectionMESurvival(ref_dirs))


class MOMapElitesArchive:
    def __init__(self, niche_count):
        self.niche_count = niche_count
        self.elites_archive = [None]*niche_count

    def update(self, pop, rank = 0):
        front = list()
        for i, individual in enumerate(pop):
            if individual.data['rank'] == rank:
                front.append((i, individual))
            elif individual.data['rank'] > rank:
                break
        niche_to_ind_mapping = {}
        for i, individual in front:
            if individual.data['niche'] not in niche_to_ind_mapping:
                niche_to_ind_mapping[individual.data['niche']] = (i, individual)
            else:
                niche_to_ind_mapping[individual.data['niche']] = (i, individual) if individual.data['dist_to_niche'] < niche_to_ind_mapping[individual.data['niche']][1].data['dist_to_niche'] else niche_to_ind_mapping[individual.data['niche']]
        niches_keys = list(niche_to_ind_mapping.keys())
        for _ in range(self.niche_count):
            niche_choice = random.choice(niches_keys)
            if self.elites_archive[niche_choice] is None:
                self.elites_archive[niche_choice] = niche_to_ind_mapping[niche_choice][1]
            else:
                if (niche_to_ind_mapping[niche_choice][1].F <= self.elites_archive[niche_choice].F).all():
                    self.elites_archive[niche_choice] = niche_to_ind_mapping[niche_choice][1]
                elif (self.elites_archive[niche_choice].F <= niche_to_ind_mapping[niche_choice][1].F).all():
                    pop[niche_to_ind_mapping[niche_choice][0]] = self.elites_archive[niche_choice]
                else:
                    choice  = random.random()
                    if choice <= 0.5:
                        self.elites_archive[niche_choice] = niche_to_ind_mapping[niche_choice][1]
                    else:
                        pop[niche_to_ind_mapping[niche_choice][0]] = self.elites_archive[niche_choice]
        return pop

    def niching_update(self, pop, n_remaining, niche_count, niche_of_individuals, dist_to_niche):
        survivors = []

        # boolean array of elements that are considered for each iteration
        mask = np.full(len(pop), True)

        while len(survivors) < n_remaining:

            # number of individuals to select in this iteration
            n_select = n_remaining - len(survivors)

            # all niches where new individuals can be assigned to and the corresponding niche count
            next_niches_list = np.unique(niche_of_individuals[mask])
            next_niche_count = niche_count[next_niches_list]

            # the minimum niche count
            min_niche_count = next_niche_count.min()

            # all niches with the minimum niche count (truncate if randomly if more niches than remaining individuals)
            next_niches = next_niches_list[np.where(next_niche_count == min_niche_count)[0]]
            next_niches = next_niches[np.random.permutation(len(next_niches))[:n_select]]

            for next_niche in next_niches:

                # indices of individuals that are considered and assign to next_niche
                next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

                # shuffle to break random tie (equal perp. dist) or select randomly
                np.random.shuffle(next_ind)

                if niche_count[next_niche] == 0:
                    
                    next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
                    if self.elites_archive[next_niche] is None:
                        self.elites_archive[next_niche] = pop[next_ind]
                    else:
                        if (pop[next_ind].F <= self.elites_archive[next_niche].F).all():
                            self.elites_archive[next_niche] = pop[next_ind]
                        elif (self.elites_archive[next_niche].F <= pop[next_ind].F).all():
                            pop[next_ind] = self.elites_archive[next_niche]
                        else:
                            choice  = random.random()
                            if choice <= 0.5:
                                self.elites_archive[next_niche] = pop[next_ind]
                            else:
                                pop[next_ind] = self.elites_archive[next_niche]


                else:
                    # already randomized through shuffling
                    next_ind = next_ind[0]
                    if self.elites_archive[next_niche] is None:
                        self.elites_archive[next_niche] = pop[next_ind]
                    else:
                        if (pop[next_ind].F <= self.elites_archive[next_niche].F).all():
                            self.elites_archive[next_niche] = pop[next_ind]
                        elif (self.elites_archive[next_niche].F <= pop[next_ind].F).all():
                            pop[next_ind] = self.elites_archive[next_niche]
                        else:
                            choice  = random.random()
                            if choice <= 0.5:
                                self.elites_archive[next_niche] = pop[next_ind]
                            else:
                                pop[next_ind] = self.elites_archive[next_niche]
                # add the selected individual to the survivors
                mask[next_ind] = False
                survivors.append(int(next_ind))

                # increase the corresponding niche count
                niche_count[next_niche] += 1

        return survivors

#Survival class helper functions
def calc_niche_count(n_niches, niche_of_individuals):
    niche_count = np.zeros(n_niches, dtype=int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count

class ReferenceDirectionMESurvival(Survival):

    def __init__(self, ref_dirs):
        super().__init__(filter_infeasible=True)
        self.ref_dirs = ref_dirs
        self.opt = None
        self.norm = HyperplaneNormalization(ref_dirs.shape[1])
        self.first_front_archive = MOMapElitesArchive(len(self.ref_dirs))
        self.last_front_archive = MOMapElitesArchive(len(self.ref_dirs))


    def _do(self, problem, pop, n_survive, D=None, **kwargs):

        # attributes to be set after the survival
        F = pop.get("F")

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        # update the hyperplane based boundary estimation
        hyp_norm = self.norm
        hyp_norm.update(F, nds=non_dominated)
        ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point

        #  consider only the population until we come to the splitting front
        #s_t
        I = np.concatenate(fronts)
        pop, rank, F = pop[I], rank[I], F[I]

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        niche_of_individuals, dist_to_niche, dist_matrix = \
            associate_to_niches(F, self.ref_dirs, ideal, nadir)

        # attributes of a population
        pop.set('rank', rank,
                'niche', niche_of_individuals,
                'dist_to_niche', dist_to_niche)
        pop = self.first_front_archive.update(pop)
        # set the optimum, first front and closest to all reference directions
        closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        self.opt = pop[intersect(fronts[0], closest)]

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=int)
            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = self.last_front_archive.niching_update(pop[last_front], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front])

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]

        return pop



