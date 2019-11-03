# ml-unsupervised-learning

1. Setup Mini Conda:

Download Conda:
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

Install conda:
sh Miniconda3-latest-Linux-x86_64.sh

2. Create  virtual env (Use the environment.yml):
conda env create --file environment.yml

3. Activate environment:
conda activate ml-raghu


4. Run the experiments using: This will run all the experiments for both the datasets(wine and bank):
PYTHONPATH=../:. python -W ignore  experiment.py



References:
1. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#)

2. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture#)

3. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#)

4. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#)

5. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/random_projection.html#)

6. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#)

7. https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

8. https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

9. https://stackoverflow.com/questions/26645642/plot-multi-dimension-cluster-to-2d-plot-python

10. https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

11.	P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009. (Retrieved from https://archive.ics.uci.edu/ml/datasets/wine+quality)
12.	[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014 (Retrieved from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

13. https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/decomposition/base.py#L135-L159