import csv
import itertools


### Non sustainable interest rate
n_sus_interest_rate = np.arange(0.01, 0.5, 0.03)
### sustainability cost multiplier
sus_cost_mult = np.linspace(-0.1, 0.3333, 4)
### natural resources recovery rate
rec_rate = np.linspace(0.5, 5.0, 4)
### entrepreneurship rate
entr_rate = 10.0**np.arange(-4, -1, 1)


with open('/Volumes/Donnees/switchdrive/Ponctuel/ModèleFlorian/Analysis/Parameters/param_space.csv', 'w') as f:
    ### Initialize csv writer
    csvwriter = csv.writer(f, delimiter=',')
    csvwriter.writerow(['non_sustainable_interest_rate', 'sustainability_cost_mult', 'resources_recovery_rate', 'entrepreneurship_rate'])
    for (nsr, cm, rr, er) in itertools.product(n_sus_interest_rate,
                                               sus_cost_mult,
                                               rec_rate,
                                               entr_rate):
        ### Write parameters
        csvwriter.writerow((nsr, cm, rr, er))


