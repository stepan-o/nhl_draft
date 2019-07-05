# Description of methodology
# Classification algorithms
# Maximum margin classification with support vector machines
From [Python Machine Learning - Second Edition By Sebastian Raschka, Vahid Mirjalili](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning-second-edition)

Another powerful and widely used learning algorithm is the Support Vector Machine (SVM), which can be considered an extension of the perceptron. Using the perceptron algorithm, we minimized misclassification errors. However, in SVMs our optimization objective is to maximize the margin. The margin is defined as the distance between the separating hyperplane (decision boundary) and the training samples that are closest to this hyperplane, which are the so-called support vectors. This is illustrated in the following figure:

<img src=img/support_vectors.jpg alt="support_vectors">

## Maximum margin intuition
The rationale behind having decision boundaries with large margins is that they tend to have a lower generalization error whereas models with small margins are more prone to overfitting. To get an idea of the margin maximization, let's take a closer look at those positive and negative hyperplanes that are parallel to the decision boundary, which can be expressed as follows:

$ w_0 + \bold{w}^T x_{pos} = 1 $
