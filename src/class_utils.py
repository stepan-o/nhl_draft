from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.plot_utils import plot_decision_regions


def fit_class(X, y, classifier='lr', xlabel="x1", ylabel="y2",
              lr_c=100.0, perc_max_iter=40, perc_eta=0.1,
              svm_l_c=1.0, svm_k_c=1.0, svm_k_gamma=0.2,
              tree_criterion='gini', tree_max_depth=4,
              forest_criterion='gini', forest_n_estimators=25, forest_njobs=2,
              knn_nn=5, knn_p=2, knn_metric='minkowski',
              random_state=1, plot_res='show', save_path=""):
    models = {
        'lr': LogisticRegression(C=lr_c, random_state=random_state),
        'perc': Perceptron(max_iter=perc_max_iter, eta0=perc_eta, random_state=random_state),
        'svm_linear': SVC(kernel='linear', C=svm_l_c, random_state=random_state),
        'svm_kernel': SVC(kernel='rbf', C=svm_k_c, gamma=svm_k_gamma, random_state=random_state),
        'tree': DecisionTreeClassifier(criterion=tree_criterion,
                                       max_depth=tree_max_depth,
                                       random_state=random_state),
        'forest': RandomForestClassifier(criterion=forest_criterion,
                                         n_estimators=forest_n_estimators,
                                         n_jobs=forest_njobs,
                                         random_state=random_state),
        'knn': KNeighborsClassifier(n_neighbors=knn_nn, p=knn_p,
                                    metric=knn_metric)
    }
    if classifier == 'all':
        for name, class_model in models.items():
            model = class_model
            print("\n----- Fitting", name.upper())
            t = time()
            model.fit(X, y)
            plot_decision_regions(X, y, classifier=model, alpha=0.3, res=plot_res, name=name,
                                  xlabel=xlabel, ylabel=ylabel, save_path=save_path,
                                  title=name.upper() + " classification algorithm, "
                                                       "\ndecision boundary"
                                                       "\nx1: {0}, x2: {1}".format(xlabel, ylabel))
            elapsed = time() - t
            print("\n - took {0:.2f} seconds.".format(elapsed))
    elif type(classifier) == list:
        for classi in classifier:
            try:
                model = models[classi]
            except (KeyError, TypeError):
                raise AttributeError("Parameter 'classifier' must be string or list of strings with one or several of "
                                     "'lr', 'perc', 'svm_linear', 'svm_kernel', 'tree', 'forest', 'knn', or 'all'")
            print("\n----- Fitting", classi.upper())
            t = time()
            model.fit(X, y)
            plot_decision_regions(X, y, classifier=model, alpha=0.3, res=plot_res, name=classi,
                                  xlabel=xlabel, ylabel=ylabel, save_path=save_path,
                                  title=classi.upper() + " classification algorithm, "
                                                         "\ndecision boundary"
                                                         "\nx1: {0}, x2: {1}".format(xlabel, ylabel))
            elapsed = time() - t
            print("\n - took {0:.2f} seconds.".format(elapsed))
    else:
        try:
            model = models[classifier]
        except (KeyError, TypeError):
            raise AttributeError("Parameter 'classifier' must be string or list of strings with one or several of "
                                 "'lr', 'perc', 'svm_linear', 'svm_kernel', 'tree', 'forest', 'knn', or 'all'")
        print("\n----- Fitting", classifier.upper())
        t = time()
        model.fit(X, y)
        plot_decision_regions(X, y, classifier=model, alpha=0.3, res=plot_res, name=classifier,
                              xlabel=xlabel, ylabel=ylabel, save_path=save_path,
                              title=classifier.upper() + " classification algorithm, "
                                                         "\ndecision boundary"
                                                         "\nx1: {0}, x2: {1}".format(xlabel, ylabel))
        elapsed = time() - t
        print("\n - took {0:.2f} seconds.".format(elapsed))
