from pymoo.factory import get_problem, get_reference_directions

from pymoo.core.callback import Callback
from pymoo.factory import get_performance_indicator
from pymoo.visualization.scatter import Scatter
import numpy as np
import json
import copy

def get_extreme_points_c(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6

    # add the old extreme points to never loose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * weights[:, None, :], axis=2)

    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points

def save_json(path, analytics):
    with open(path + ".json", 'a') as fp:           
        fp.write('\n')
        json.dump(analytics, fp)

def mat_to_arr(mat):
    arr = []
    for vec in mat:
        arr += [list(vec)]
    return arr

class Analytics(Callback):
    def __init__(self, n_dim):
        super().__init__()
        self.front_history = {"fronts" : [], "hyper_volumes" : []}
        self.ideal_point = np.full(n_dim, np.inf)
        self.worst_point = np.full(n_dim, -np.inf)
        self.extreme_points = None
    
    def notify(self, algorithm):
        first_front = np.array([point.F for point in algorithm.opt])
        self.ideal_point = np.min(np.vstack((self.ideal_point, first_front)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, first_front)), axis=0)
        epsilon = np.zeros(shape=self.worst_point.shape)
        epsilon.fill(0.1)
        ref_point = self.worst_point + epsilon
        self.front_history["fronts"].append(first_front)
        hv = get_performance_indicator("hv", ref_point=ref_point)
        self.extreme_points = get_extreme_points_c(first_front, self.ideal_point, self.extreme_points)
        hypervolume = hv.do(np.vstack((self.extreme_points, self.ideal_point)))
        self.front_history["hyper_volumes"].append(hypervolume)


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

def matr_add_diff_size(a, b):
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c

def extract_median_front(fronts, hyper_volumes):
    hyper_volumes_copy = [(i, hv) for i, hv in enumerate(hyper_volumes)]
    hyper_volumes_copy.sort(key = lambda x : x[1])
    if len(hyper_volumes_copy)%2 == 0:
        return (matr_add_diff_size(fronts[len(hyper_volumes_copy)//2], fronts[len(hyper_volumes_copy)//2 - 1])) /2.
    else:
        return fronts[len(hyper_volumes_copy)//2]