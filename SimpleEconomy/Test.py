

import numpy as np
import matplotlib.pyplot as plt
import time
import gc

from SimpleEconomy import Economy


eco = Economy(Max_Resource = 10_000,  Res_Recovery_Rate = 1.0, 
              Taux_Entrep = 0.001,
              Sustainability_Interest_Rate_Mult = 1.0,
              Prod_Adjust = 0.3,
              Init_Wealth = 1000,
              Sustainability_Cost_Mult = 0.005,
              N_Firm = 100, tol = 10.0**-6)

T_Max = 10000

resource = np.zeros(T_Max)
n_firms_entry = np.zeros(T_Max)
n_firms = np.zeros(T_Max)
bank_profit = np.zeros(T_Max)
price = np.zeros(T_Max)
prod_sustainability = np.zeros(T_Max)
gdp = np.zeros(T_Max)
tot_firms_assets = np.zeros(T_Max)

t0 = time.time()

gc.collect(generation=2)
gc.collect(generation=1)

for t in range(T_Max):
    n_firms_entry[t] = eco.firm_enter()
    eco.firm_decide_production()
    eco.firm_pay_dividend()
    prod_sustainability[t] = eco.average_production_sustainability()
    eco.firm_get_credit()
    gdp[t] = eco.Avail_Wealth
    price[t] = eco.firm_produce_sell()
    eco.firm_pay_salary()
    bank_profit[t] = eco.firm_pay_bank()
    n_firms[t] = eco.firm_exit()
    resource[t] = eco.recover_resource()
    n_firms[t] = eco.firms_count()
    tot_firms_assets[t] = np.sum(eco.Firm_Avoir[eco.Firm_Existe])
    eco.increment_iteration()
    if resource[t] < 0:
        print(f'No more available resources at iteration {t}')
        break

### Est-ce-que les firmes doivent verser des dividendes ?

plt.plot(prod_sustainability[:t])
plt.plot(resource[:t])
plt.plot(bank_profit[:t])
plt.plot(gdp[10:t])
plt.plot(tot_firms_assets[:t])
plt.hist(eco.Firm_Sustainability[eco.Firm_Existe], bins = 50)
plt.scatter(np.arange(100, t), price[100:t])
