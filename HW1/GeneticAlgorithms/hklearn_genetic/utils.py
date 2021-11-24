import numpy as np

class ProblemUtils:
    @staticmethod
    def _to_matrix(X):
        new_X = np.array(X[0].genotype)
        for i in range(1, len(X)):
            new_X = np.concatenate((new_X, np.array(X[i].genotype)))
        new_X = new_X.reshape((len(X), len(X[0].genotype)))
        return new_X
    @staticmethod
    def _to_genotypes(X, X_mat):
        for i in range(len(X)):
            X[i].genotype = list(X_mat[i,:])
        return X

    @staticmethod
    def _to_evaluated(X, X_mat):
        for i in range(len(X)):
            X[i].fitness_metric = X_mat[i]
        return X

    @staticmethod
    def _to_evaluated_matrix(X):
        li = list(X[0].phenotype) + [-X[0].fitness_metric]
        mat_X = np.array(li)
        for i in range(1, len(X)):
            mat_X = np.concatenate((mat_X, np.array(list(X[i].phenotype) + [-X[i].fitness_metric])))
        mat_X = mat_X.reshape((len(X), len(X[0].phenotype) + 1))
        li = [list(X[0].phenotype) + [-X[0].fitness_metric]]
        for i in range(1, len(X)):
            li.append(list(X[i].phenotype) + [-X[i].fitness_metric])
        return li, mat_X