import numpy as np
import csv
import sys
import scipy

from SimpleEconomy import Economy

### Row to read in parameters file
param_row = sys.argv[1] ## For testing: param_row = 10

## Path to  file
params_file_path = '/users/fbarras3/sustainable_finance_model/param_space.csv'
#params_file_path = '/Volumes/Donnees/switchdrive/Ponctuel/ModèleFlorian/Analysis/Parameters/param_space.csv'

### Load parameters
with open(params_file_path, newline='\n') as csvfile:
    paramreader = csv.reader(csvfile, delimiter=',')
    header = None
    values = None
    for row in paramreader:
        if paramreader.line_num == 1:
            header = row
        elif paramreader.line_num == param_row+1:
            values = row
        
        ### Break if needed row has been read
        if paramreader.line_num >= param_row+1:
                break

### Model parameters
run_params = {k: eval(v) for (k, v) in zip(header, values)}

#### Variables to measure
#gdp
#average sustainability
#resources level
#correlation between asset and sustainability

#### Number of maximum iterations
T_max = 15_000

T_resources_exhausted = 10**9

### Results header to write in csv file
results_headers = ['run_index'] +\
    header +\
    ['t_res_exhausted'] +\
    [f'gdp_{i}' for i in range(0, T_max, 500)] + ['gdp_t_re'] +\
    [f'average_sustainability_{i}' for i in range(0, T_max, 500)] + ['average_sustainability_t_re'] +\
    [f'resources_{i}' for i in range(0, T_max, 500)] + ['resources_t_re'] +\
    [f'corr_asset_sustainability_{i}' for i in range(0, T_max, 500)] + ['corr_asset_sustainability_t_re']

#### Results output
output_path = f'/scratch/fbarras3/sustainable_finance_model_output/results_params_ind_{sys.argv[1]}.csv'
#output_path = '/Volumes/Donnees/switchdrive/Ponctuel/ModèleFlorian/Analysis/Output/test.csv'

### Write header in result file
with open(output_path, 'w') as f:
    ### Initialize csv writer
    csvwriter = csv.writer(f, delimiter=',')
    ### Write parameters
    csvwriter.writerow(results_headers)


#### Run N times
for i in range(10):
    ### Variables to measure
    gdp = np.zeros(T_max)
    average_sustainability = np.zeros(T_max)
    resources_level = np.zeros(T_max)
    corr_asset_sustainability = np.zeros(T_max)

    ## Initialize the economy
    eco = Economy(Max_Resource = 10_000, 
                  Res_Recovery_Rate = run_params['resources_recovery_rate'],
                  Taux_Entrep = run_params['entrepreneurship_rate'],
                  Sustainability_Interest_Rate_Mult = run_params['non_sustainable_interest_rate'],
                  sus_lvl_sens = 1.0,
                  Prod_Adjust = 0.3,
                  Init_Wealth = 1000,
                  Sustainability_Cost_Mult = run_params['sustainability_cost_mult'],
                  N_Firm = 100, tol = 10.0**-6)
    

    print(f'Starting run {i}.')

    # Run iterations
    for t in range(T_max):
        eco.firm_enter()
        gdp[t] = eco.firm_decide_production()
        eco.firm_pay_dividend()
        average_sustainability[t] = eco.average_production_sustainability()
        
        eco.firm_get_credit()
        gdp[t] *= eco.firm_produce_sell()
        eco.firm_pay_salary()

        eco.firm_pay_bank()
        eco.firm_exit()
        resources_level[t] = eco.recover_resource()
        corr_asset_sustainability[t] = scipy.stats.kendalltau(eco.Firm_Avoir[eco.Firm_Existe], eco.Firm_Sustainability[eco.Firm_Existe]).correlation
        eco.increment_iteration()
        if resources_level[t] < 0:
            T_resources_exhausted = t
            break

    #### Save results in csv
    results_values = [i+1] +\
        values +\
        [T_resources_exhausted] +\
        [gdp[i] if i <= T_resources_exhausted else None for i in range(0, T_max, 500)] + [gdp[min(T_resources_exhausted, T_max-1)]] +\
        [average_sustainability[i] if i <= T_resources_exhausted > 0.0 else None for i in range(0, T_max, 500)] + [average_sustainability[min(T_resources_exhausted, T_max-1)]] +\
        [resources_level[i] if i <= T_resources_exhausted > 0.0 else None for i in range(0, T_max, 500)] + [resources_level[min(T_resources_exhausted, T_max-1)]] +\
        [corr_asset_sustainability[i] if i <= T_resources_exhausted > 0.0 else None for i in range(0, T_max, 500)] + [corr_asset_sustainability[min(T_resources_exhausted, T_max-1)]]    
    
    ### Write header in result file
    with open(output_path, 'a') as f:
        ### Initialize csv writer
        csvwriter = csv.writer(f, delimiter=',')
        ### Write parameters
        csvwriter.writerow(results_values)
