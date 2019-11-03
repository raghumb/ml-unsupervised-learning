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
1. 

10.	P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009. (Retrieved from https://archive.ics.uci.edu/ml/datasets/wine+quality)
11.	[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014 (Retrieved from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
