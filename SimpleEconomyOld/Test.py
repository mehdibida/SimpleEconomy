from SimpleEconomy import Economy

import numpy as np
import matplotlib.pyplot as plt
import gc
import time

###Checker si les crédits doivent ou pas aller dans les fonds avoirs et le faire le cas échéant.

##Pq ça bloque avec des coûts fixes à zéro?

from SimpleEconomy import Economy

np.seterr(all='raise')

T_Max = 1400

t_invest_val = [0.1]
tif = 0
n_dur_fin = np.zeros(len(t_invest_val))
n_ndur_fin = np.zeros(len(t_invest_val))

q_dur_fin = np.zeros(len(t_invest_val))
q_ndur_fin = np.zeros(len(t_invest_val))

np.array([i for i in range(6)])[1:]


for tif in range(len(t_invest_val)):

	eco = Economy(Max_Resource= 0.2 * 10.0**7, Res_Recovery_Rate= .5, Taux_Entrep= 0.005, Taux_Interet= 0.07, Firm_Fixed_Cost= .5, Prime_D = 0.1, Firm_Markup = 0.1, 
			      Labor_Rigidity=0.0, Prod_Adjust = 0.333, Taux_Invest_Fonds= t_invest_val[tif], Init_Wealth= 500.0, Sens_Prix = 1.0, N_Firm = T_Max)

	resource_level = np.zeros(T_Max)
	gdp = np.zeros(T_Max)
	total_money = np.zeros(T_Max)
	total_avail_wlth = np.zeros(T_Max)
	total_firm_avoir = np.zeros(T_Max)
	income = np.zeros(T_Max)
	n_firms_durable = np.zeros(T_Max)
	n_firms = np.zeros(T_Max)
	n_firms_exit = np.zeros(T_Max)
	n_firms_entry = np.zeros(T_Max)
	fonds_avoir = np.zeros(T_Max)
	salaire = np.zeros(T_Max)

	sus_consumed_quantity = np.zeros(T_Max)
	n_sus_consumed_quantity = np.zeros(T_Max)
	n_sus_prod_quantity = np.zeros(T_Max)
	n_sus_prod_quantity_new = np.zeros(T_Max)

	prod_fac = 1.0

	#t0 = time.time()

	gc.collect(generation=2)
	gc.collect(generation=1)

	t_min_durable = 0.0
	for t in range(T_Max):
		if t % 1 == 0: print("Step:", t, ";")
		if ((t/T_Max) >= t_min_durable) & (t % 1 == 0):
			print("Prop durable change to : ", min(9.0 * (t / T_Max - t_min_durable), 1.0))
			#eco.set_prop_durable(min(9.0 * (t / T_Max - t_min_durable), 1.0))
			eco.set_res_threshold(1.0 * eco.Max_Resource)
		n_firms_entry[t] = eco.firm_enter(prob_durable=.00, scale = 2.0 * eco.Firm_Fixed_Cost)
		eco.firm_decide_quantity()
		eco.reinitialize_income()
		#eco.firm_budgetize_and_pay_shareholders(prop_cost_keep=0.1) ##income ++
		eco.firm_budgetize_and_pay_shareholders_dynamic(min_prop_cost_keep=0.0) ##income ++
		#print(eco.Firm_Production[eco.Firm_Existe])
		#time.sleep(0.1)
		#print(eco.Firm_Production[eco.Firm_Existe])
		#print("---------Step ", t, " ----------------")
		eco.firm_decide_credit()
		credit_dem_base = np.copy(eco.Firm_Credit_Dem)
		#eco.Firm_Credit_Dem[eco.Firm_Existe]
		#eco.Fonds_Avoir = 0.04
		eco.firm_credit_dem_fund()
		#eco.Firm_Credit_Rec_Fonds[eco.Firm_Existe]
		eco.firm_credit_dem_bank(sens_new_info= 0.5, prob_transform= lambda x : x**1.0)
		#print("Credit rec.:", eco.Firm_Credit_Rec_Bank[eco.Firm_Existe])
		#time.sleep(0.5)
		#input()
		n_sus_prod_quantity[t] = eco.Resource
		eco.firm_produce()
		resource_level[t] = eco.Resource
		n_sus_prod_quantity[t] -= eco.Resource
		n_sus_prod_quantity_new[t] = sum(eco.Firm_Production[(eco.Firm_Moment_Creation >= (t-10)) & 
						    							     np.logical_not(eco.Firm_Durable) &
															 eco.Firm_Existe])

		#eco.Firm_Production[eco.Firm_Existe] *= prod_fac
		#eco.Firm_Last_Production[eco.Firm_Existe] *= prod_fac
		#prod_fac += 0.0

		#print("Firm Rec Fonds: ", eco.Firm_Credit_Rec_Fonds[eco.Firm_Existe])
		eco.firm_set_price()
		#print("Avoir.:",eco.Firm_Avoir[eco.Firm_Existe])
		#input()
		eco.firm_pay_salary() ##income ++
		eco.cons_invest_fund() ## wealth --
		eco.consume_max_wealth(prop_cons= .5) ## wealth --
		eco.firm_pay_fund() ## fonds ++
		eco.firm_pay_bank() ## wealth ++, but not income
		eco.cons_withdraw_fund(sens_new_info=0.5)
		salaire[t] = eco.update_salary(0.7)
		eco.recover_resource()
		n_firms_exit[t] = eco.firm_exit()
		eco.increment_iteration()
		#input()
		#print(eco.Firm_Avoir)
		#print(eco.Firm_Avoir)
		gdp[t] = eco.gross_product()
		total_firm_avoir[t] = np.sum(eco.Firm_Avoir[eco.Firm_Existe])
		n_firms[t] = eco.nb_firms()
		n_firms_durable[t] = eco.nb_sustainable_firms()
		sus_consumed_quantity[t] = eco.sustainable_consumed_quantity()
		n_sus_consumed_quantity[t] = eco.non_sustainable_consumed_quantity()
		fonds_avoir[t] = eco.Fonds_Avoir
		total_money[t] = eco.total_money()
		income[t] = eco.Income
		total_avail_wlth[t] = eco.Avail_Wealth


	n_dur_fin[tif] = eco.nb_sustainable_firms()
	n_ndur_fin[tif] = eco.nb_firms() - n_dur_fin[tif]
	q_dur_fin[tif] = eco.sustainable_consumed_quantity()
	q_ndur_fin[tif] = eco.non_sustainable_consumed_quantity()


#plt.plot(t_invest_val, n_ndur_fin, c= 'red'); plt.plot(t_invest_val, n_dur_fin, c = 'green'); plt.show()
#plt.plot(t_invest_val, q_ndur_fin, c= 'red'); plt.plot(t_invest_val, q_dur_fin, c = 'green'); plt.show()


#eco.Firm_Production[eco.Firm_Existe]
#eco.Firm_Productivite[eco.Firm_Existe]
#eco.Firm_Avoir[eco.Firm_Existe]

#t1 = time.time()
#t1 - t0

##ajouter des épargnants qui veulent maximiser leur rémunération par leur épargne, épargne investie
plt.plot(np.log(gdp[gdp > 0] / salaire[gdp > 0])); plt.plot(np.log(gdp[gdp > 0])); plt.show();

plt.plot(gdp)

#plt.hist(eco.Firm_Avoir[eco.Firm_Existe])

plt.plot(total_avail_wlth / total_money, c = 'orange'); plt.plot(fonds_avoir / total_money, c = 'purple'); plt.show()

plt.plot(total_avail_wlth); plt.plot(income);

plt.plot(fonds_avoir) ; plt.show();

plt.plot(total_avail_wlth * eco.Taux_Entrep); plt.show();

plt.plot(resource_level) ; plt.show();

plt.plot(resource_level * eco.Res_Recovery_Rate * (1.0 - resource_level / eco.Max_Resource));
plt.plot(n_sus_prod_quantity, c = 'red'); plt.plot(n_sus_prod_quantity_new, c = 'green');
plt.plot(n_sus_prod_quantity + n_sus_prod_quantity_new, c = 'blue');

plt.plot(np.diff(n_sus_prod_quantity[:370] + n_sus_prod_quantity_new[:370]), c = 'blue');


plt.plot(n_sus_prod_quantity_new);
plt.show();

plt.plot(n_firms - n_firms_durable, c= 'red'); plt.plot(n_firms_durable, c = 'green'); plt.show()

plt.plot(n_sus_consumed_quantity, c = 'red'); plt.plot(sus_consumed_quantity, c = 'green'); plt.show();
plt.plot(sus_consumed_quantity / (n_sus_consumed_quantity + sus_consumed_quantity), c = 'green'); plt.show()

plt.scatter(eco.Firm_Moment_Creation[eco.Firm_Existe], eco.Firm_Avoir[eco.Firm_Existe], c = np.where(eco.Firm_Durable[eco.Firm_Existe], 'green', 'red')); plt.show()



plt.plot(total_avail_wlth / total_money)

plt.plot(total_avail_wlth)

plt.scatter(eco.Firm_Moment_Creation[eco.Firm_Existe], eco.Firm_Last_Production[eco.Firm_Existe], c = np.where(eco.Firm_Durable[eco.Firm_Existe], 'green', 'red')); plt.show()

plt.scatter(eco.Firm_Moment_Creation[eco.Firm_Existe], eco.Firm_Prix[eco.Firm_Existe], c = np.where(eco.Firm_Durable[eco.Firm_Existe], 'green', 'red')); plt.show()

plt.scatter(eco.Firm_Moment_Creation[eco.Firm_Existe], eco.Firm_Credit_Miss_Dem_Ratio[eco.Firm_Existe], c = np.where(eco.Firm_Durable[eco.Firm_Existe], 'green', 'red'))

plt.loglog(fonds_avoir); plt.show()

plt.loglog(np.sort(eco.Firm_Last_Production[eco.Firm_Existe])[::-1])

plt.plot(np.sort(eco.Firm_Last_Production[eco.Firm_Existe])[::-1])

plt.plot(np.log(salaire[salaire > 0]))
eco.Prev_Last_Total_Prod
eco.Last_Total_Prod


plt.plot(n_firms_exit);
plt.plot(n_firms_entry);

plt.plot(n_firms);
plt.plot((n_firms_entry / n_firms)[10::]);
plt.plot(np.diff(n_firms) / n_firms[1::]);


np.seterr(all='print')



eco.Firm_Credit_Dem[eco.Firm_Existe & (eco.Firm_Credit_Dem < eco.tol)] = np.Inf
eco.Firm_Credit_Dem[eco.Firm_Existe]


firm_cmdr = eco.Firm_Existe & (eco.Firm_Credit_Score < np.Inf)

credit_dem_base[firm_cmdr]
eco.Firm_Credit_Miss_Dem_Ratio[firm_cmdr]#[[27, 78]]
eco.Firm_Credit_Score[firm_cmdr]#[[27, 78]]
eco.Firm_Credit_Dem[firm_cmdr]#[[27, 78]]
eco.Firm_Credit_Rec_Bank[firm_cmdr]#[[27, 78]]
eco.Firm_Credit_Rec_Fonds[firm_cmdr]#[[27, 78]]

eco.Firm_Production[firm_cmdr]

eco.Firm_Moment_Creation[firm_cmdr][3136]

eco.Firm_Production[firm_cmdr][3136]
eco.Firm_Credit_Rec_Bank[firm_cmdr][3136]
eco.Firm_Durable[firm_cmdr][3136]
eco.Firm_Credit_Rec_Fonds[firm_cmdr][3136]
eco.Firm_Credit_Dem[firm_cmdr][3136]

eco.Firm_Last_Production[firm_cmdr]

eco.Firm_Durable[firm_cmdr]
eco.Firm_New_Durable[firm_cmdr]

eco.Firm_Credit_Dem[firm_cmdr] / (eco.Firm_Credit_Dem[firm_cmdr] + eco.Firm_Credit_Rec_Bank[firm_cmdr] + eco.Firm_Credit_Rec_Fonds[firm_cmdr])

np.where((eco.Firm_Credit_Dem[firm_cmdr] + eco.Firm_Credit_Rec_Bank[firm_cmdr] + eco.Firm_Credit_Rec_Fonds[firm_cmdr]) == 0.0)

np.where(eco.Firm_Credit_Dem[firm_cmdr] + eco.Firm_Credit_Rec_Bank[firm_cmdr] + eco.Firm_Credit_Rec_Fonds[firm_cmdr] == 0.0)

eco.Firm_Credit_Score[eco.Firm_Durable]


eco.Income

eco.Fonds_Avoir

sum(eco.Firm_Avoir[eco.Firm_Existe])

#plt.plot(eco.Firm_Production[eco.Firm_Existe]); plt.show()


x_t+1 = (1-a) x_t + a v

a^12 = 0.

self.Resource + self.Res_Recovery_Rate * self.Resource * (1.0 - self.Resource / self.Max_Resource)

m = (np.sin(np.arange(0., 100., 0.3)) + 1)
p = np.ones(len(m)) * 0.0

th = 0.65

for i in range(1, len(m)):
	p[i] = th * p[i-1] + (1. - th) * 10.0

plt.plot(p); #plt.plot(m);
p[0:20]

x = np.arange(0, 1, 0.01)
r = 0.3
plt.plot(x, x * r * (1 - x/1) / (1 - x))

0.7**2

a = np.ones(10)

b = np.zeros(10)

a[0:4] = b[0:4]

b[0:4] = 10.0

a
