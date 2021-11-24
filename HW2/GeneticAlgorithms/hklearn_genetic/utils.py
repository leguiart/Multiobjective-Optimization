
import numpy as np
import matplotlib.pyplot as plt

class ProblemUtils:
    @staticmethod
    def _to_matrix_genotypes(X):

        new_X = np.array(X[0].genotype)

        for i in range(1, len(X)):
            new_X = np.concatenate((new_X, np.array(X[i].genotype)))
        new_X = new_X.reshape((len(X), len(X[0].genotype)))
        return new_X

    @staticmethod
    def _to_matrix_phenotypes(X):

        new_X = np.array(X[0].phenotype)

        for i in range(1, len(X)):
            new_X = np.concatenate((new_X, np.array(X[i].phenotype)))
        new_X = new_X.reshape((len(X), len(X[0].phenotype)))
        return new_X

    @staticmethod
    def _to_matrix_fitness(X):
        new_X = np.array(X[0].fitness_metric)
        for i in range(1, len(X)):
            new_X = np.concatenate((new_X, np.array(X[i].fitness_metric)))
        new_X = new_X.reshape((len(X), len(X[0].fitness_metric)))
        return new_X
    
    @staticmethod
    def _to_genotypes(X, X_mat):
        for i in range(len(X)):
            X[i].genotype = list(X_mat[i,:])
        return X

    @staticmethod
    def _to_evaluated(X, X_mat):
        # print(X_mat.shape)
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

class PlotUtils:
    @staticmethod
    def plot_mat_2d_lines(mats, title, sup_title, play = False):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        fig.suptitle(sup_title, fontsize=16)
        ax = fig.add_subplot(111)
        ax.set_title(title)
        for j in range(len(mats[1])):
            x = [0, mats[1][j, 0]]
            y = [0, mats[1][j, 1]]
            #ax.scatter(x,y,z)
            ax.plot(x, y, color='red')

        for mat in mats:
            xs = mat[:, 0]
            ys = mat[:, 1]
            ax.scatter(xs,ys, s=10)
        if play:
            plt.show(block = False)
            # plt.show()
            plt.pause(0.45)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_mat_2d(mats, title, sup_title, labels = [], play = False):
        # plt.rcParams["figure.figsize"] = [7.50, 3.50]
        # plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        fig.suptitle(sup_title, fontsize=16)
        ax = fig.add_subplot(111)
        ax.set_title(title)

        for i, mat in enumerate(mats):
            if len(labels) != 0 and mat.shape[0] == 1:
                ax.scatter(mat[:,0], mat[:,1])
                ax.annotate(labels[i], (mat[:,0], mat[:,1]))
            elif len(labels) != 0:
                xs = mat[:, 0]
                ys = mat[:, 1]
                ax.scatter(xs, ys, label = labels[i] )

            else:
                xs = mat[:, 0]
                ys = mat[:, 1]
                ax.scatter(xs, ys)

        if play:
            plt.show(block = False)
            # plt.show()
            plt.pause(0.45)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_mat_3d_lines(mats, title, sup_title, play = False):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        fig.suptitle(sup_title, fontsize=16)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        for j in range(len(mats[1])):
            x = [0, mats[1][j, 0]]
            y = [0, mats[1][j, 1]]
            z = [0, mats[1][j, 2]]
            #ax.scatter(x,y,z)
            ax.plot(x, y, z, color='red')

        for mat in mats:
            xs = mat[:, 0]

            ys = mat[:, 1]

            zs = mat[:, 2]

            ax.scatter(xs,ys,zs, s=10)
        if play:
            plt.show(block = False)
            # plt.show()
            plt.pause(0.45)
            plt.close()
        else:
            plt.show()

    def plot_mat_3d(mats, title, sup_title, play = False):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        fig.suptitle(sup_title, fontsize=16)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)

        for mat in mats:
            xs = mat[:, 0]

            ys = mat[:, 1]

            zs = mat[:, 2]

            ax.scatter(xs,ys,zs)
        if play:
            plt.show(block = False)
            # plt.show()
            plt.pause(0.45)
            plt.close()
        else:
            plt.show()