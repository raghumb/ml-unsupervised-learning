from experiments.KMeansExp import KMeansExp
from experiments.EMExp import EMExp
from experiments.SVDExp import SVDExp
from experiments.PCAExp import PCAExp
from experiments.ANNExp import ANNExp
from experiments.RandomProjExp import RandomProjExp
from DataSplitter import DataSplitter
from BankDataReader import BankDataReader
from WineDataReader import WineDataReader
import numpy as np

def experiment_invoke(dataset = 'bank'):
    if dataset == 'wine':
        reader = WineDataReader()
    else:
        reader = BankDataReader()

    ds = DataSplitter(reader)
    ds.read_split_data()
    run_final = False
    er = KMeansExp(reader, ds, run_final)
    er.experiment()

    er = EMExp(reader, ds, run_final)
    er.experiment()   
    er = ANNExp(reader, ds)
    er.experiment_bank()
    er.experiment_clusters()



    run_final = True
    er = KMeansExp(reader, ds, run_final)
    er.experiment()

    er = EMExp(reader, ds, run_final)
    er.experiment()   
    er = ANNExp(reader, ds)
    er.experiment_bank()
    er.experiment_clusters()   



experiment_invoke(dataset = 'bank')
experiment_invoke(dataset = 'wine')