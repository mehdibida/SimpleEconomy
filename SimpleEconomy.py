# Modèle : est-ce que les banques ont une influence sur la société 
import numpy as np
import matplotlib.pyplot as plt
import gc


import time

#####Comprendre pourquoi durable sans markup change les résultats


#0 System_Sustainable_Firms #nombre d'entreprise durables, initialement zero
#1 System_PIB #PIB initial, l'économie commence à 0.
#2 System_Money sum(np.maximum(Firm_Avoir[Firm_Existe], 0)) + Income #somme monétaire , l'économie commence à 0.
#3 System_Credit sum(Firm_Credit_Rec[Firm_Existe]) #nombre de crédit, personne n'a de crédit initialement.
class Economy:
	def __init__(self, Taux_Interet, Firm_Fixed_Cost, Prime_D, Firm_Markup, Prod_Ajust, Taux_Invest_Fonds, Init_Wealth, Sens_Prix, Init_Prop_Durable=0.0, N_Firm = 100, tol = 10.0**-6) -> None:
		self.Prop_Durable = Init_Prop_Durable  #Proportion minimum de prêt durable, initialement à zero
		self.Taux_Interet = Taux_Interet #Le taux d'intérêt des Fonds_Avoirs/crédits
		
		self.Income = 0.0 ##Income received during iteration

		self.Last_Total_Consumption = 0.0
		
		#self.Qualité_Durable = 2.0 #Qualité des produits vendus par les firmes durables, qualité des non-durables à 1
		#Salaires
		self.Prime_D = Prime_D #Prime pour les salaires durables (facteur) #Il est en hashtag dans le modèle V1 : pourquoi ?
		self.Salaire = 1.0 #Salaire non durable
		self.Firm_Fixed_Cost = Firm_Fixed_Cost
		self.Avail_Wealth = Init_Wealth
		self.Firm_Markup = Firm_Markup
		self.Taux_Invest_Fonds = Taux_Invest_Fonds
		self.Sens_Prix = Sens_Prix
		self.Prod_Ajust = Prod_Ajust
		
		##Current iteration of the economy
		self.Iteration = 0

		##Numerical tolerance of the model (in order to avoid errors due to floating point representation)
		self.tol = tol

		##Caracteristiques des entreprises
		self.Firm_Existe = np.full(N_Firm, False) ##L'entreprise est active.
		self.Firm_Durable = np.full(N_Firm, False) ##Durable ou pas.
		self.Firm_New_Durable = np.full(N_Firm, False) ##Firmes devenant durables.
		self.Firm_Last_Production = np.full(N_Firm, 0.0) ## Productivité des entreprises, détermine le coût

		self.Firm_Moment_Creation = np.zeros(N_Firm) ##Iteration at which firm was created

		self.Firm_Production = np.zeros(N_Firm) #Production de chaque entreprise
		self.Firm_Cost = np.zeros(N_Firm) #Niveau des profits faits par l'entreprise
		self.Firm_Avoir = np.zeros(N_Firm) #Avoir de l'entreprise
		self.Firm_Credit_Dem = np.zeros(N_Firm) #Credits demandés
		self.Firm_Credit_Rec_Bank = np.zeros(N_Firm) #Credits reçu par l'entreprise (octroyé par la banque)
		self.Firm_Credit_Rec_Fonds = np.zeros(N_Firm)
		self.Firm_Credit_Miss_Dem_Ratio = np.zeros(N_Firm) #Ratio entre les fonds reçus et les fonds demandés
		self.Firm_Demande = np.zeros(N_Firm) #Quantités demandées à chacune de entreprises
		self.Firm_Prix = np.ones(N_Firm) #Prix des entreprises

		self.Prob_Become_Durable = np.zeros(N_Firm)

		##Caractéristiques des fonds
		self.Fonds_Avoir = 0.0  #Taille du fonds (initialization à zero)
		self.Fonds_Yield = 0.0

		###Random number generator
		self.rng = np.random.default_rng()

	def set_prop_durable(self, prop) -> None:
		self.Prop_Durable = prop

	def reinitialize_income(self) -> None:
		self.Income = 0.0

	def firm_enter(self, N_Entrants, prob_durable=0.0) -> None:
		##Extend vectors if there is no sufficient place for all active firms
		if N_Entrants > (len(self.Firm_Existe) - sum(self.Firm_Existe)):
			extension_size = N_Entrants - sum(np.logical_not(self.Firm_Existe))
			self.Firm_Existe.resize(len(self.Firm_Existe) + extension_size, refcheck = False)
			self.Firm_Durable.resize(len(self.Firm_Durable) + extension_size, refcheck = False)
			self.Firm_Last_Production.resize(len(self.Firm_Last_Production) + extension_size, refcheck = False)
			self.Firm_New_Durable.resize(len(self.Firm_New_Durable) + extension_size, refcheck = False)
			self.Firm_Production.resize(len(self.Firm_Production) + extension_size, refcheck = False)
			self.Firm_Cost.resize(len(self.Firm_Cost) + extension_size, refcheck = False)
			self.Firm_Avoir.resize(len(self.Firm_Avoir) + extension_size, refcheck = False)
			self.Firm_Credit_Dem.resize(len(self.Firm_Credit_Dem) + extension_size, refcheck = False)
			self.Firm_Credit_Rec_Bank.resize(len(self.Firm_Credit_Rec_Bank) + extension_size, refcheck = False)
			self.Firm_Credit_Rec_Fonds.resize(len(self.Firm_Credit_Rec_Fonds) + extension_size, refcheck = False)
			self.Firm_Credit_Miss_Dem_Ratio.resize(len(self.Firm_Credit_Miss_Dem_Ratio) + extension_size, refcheck = False)
			self.Firm_Demande.resize(len(self.Firm_Demande) + extension_size, refcheck = False)
			self.Firm_Prix.resize(len(self.Firm_Prix) + extension_size, refcheck = False)
			self.Firm_Moment_Creation.resize(len(self.Firm_Moment_Creation) + extension_size, refcheck = False)
			self.Prob_Become_Durable.resize(len(self.Prob_Become_Durable) + extension_size, refcheck = False)

			self.Firm_Prix[np.logical_not(self.Firm_Existe)] = 1.0
		###Initialize new firms
		#Liste d'indices dispos qui correspondent aux cases qui vont contenir les nouvelles entreprises
		Indices_Entrants = np.flatnonzero(np.logical_not(self.Firm_Existe))[:N_Entrants] ##Trouver les indices qui son False
		self.Firm_Last_Production[Indices_Entrants] = 0.0
		self.Firm_Production[Indices_Entrants] = 0.0
		self.Firm_Cost[Indices_Entrants] = 0.0
		self.Firm_Avoir[Indices_Entrants] = 0.0
		self.Firm_Credit_Dem[Indices_Entrants] = 0.0
		self.Firm_Credit_Rec_Bank[Indices_Entrants] = 0.0
		self.Firm_Credit_Rec_Fonds[Indices_Entrants] = 0.0
		self.Firm_Credit_Miss_Dem_Ratio[Indices_Entrants] = 0.0
		self.Firm_Durable[Indices_Entrants] = self.rng.binomial(1, prob_durable, N_Entrants) == 1
		self.Firm_Existe[Indices_Entrants] = True
		self.Firm_Moment_Creation[Indices_Entrants] = self.Iteration
		self.Prob_Become_Durable[Indices_Entrants] = 0.0

		
	def firm_decide_quantity(self):
		new_entrants = np.flatnonzero(self.Firm_Existe & (self.Firm_Moment_Creation == self.Iteration))
		incumbents = np.flatnonzero(self.Firm_Existe & (self.Firm_Moment_Creation < self.Iteration))
		
		if len(incumbents) > 0:
			 ##2.b : Les entreprises existantes ajustent la production par rapport aux ventes précédentes. Prod_Sensibilite entre 0 et 1 et correspond à la sensibilité des entrepeneurs.
			self.Firm_Production[incumbents] = (self.Firm_Last_Production[incumbents] - self.Firm_Production[incumbents]) * \
											   (1.0 + self.Prod_Ajust * np.random.uniform(0.0, 1.0, len(incumbents)))
			
			#self.Prod_Ajust * (self.Firm_Demande[incumbents] - self.Firm_Production[incumbents])
											   #np.random.normal(0.0, 0.05 * self.Firm_Production[incumbents])
		
		if len(new_entrants) > 0:
			if len(incumbents) > 0:
				self.Firm_Production[new_entrants] = np.mean(self.Firm_Production[incumbents]) * np.random.uniform(0.8, 1.2, size = len(new_entrants))
			else:
				self.Firm_Production[new_entrants] = np.random.uniform(0.0, 1.0, size = len(new_entrants)) #* self.Avail_Wealth / self.Salaire / len(new_entrants)
		
		##Initialisation des coûts aux coûts fixes
		self.Firm_Cost[self.Firm_Existe] = self.Firm_Fixed_Cost


	def firm_decide_credit(self) -> None:
		self.Firm_Credit_Dem[self.Firm_Existe] = np.maximum(self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * \
						      								self.Firm_Production[self.Firm_Existe] - \
															self.Firm_Avoir[self.Firm_Existe], 0.0)
		

	def firm_credit_dem_fund(self) -> None:
		### Initialisation
		#Fonds  : "invetsissement" correspond à la qté totale de fonds disponible dans le fonds.
		if self.Fonds_Avoir >= sum(self.Firm_Credit_Dem[self.Firm_Existe]):
			self.Firm_Credit_Rec_Fonds[self.Firm_Existe] = self.Firm_Credit_Dem[self.Firm_Existe]
			self.Firm_Credit_Dem[self.Firm_Existe] = 0.0
		else: 
			self.Firm_Credit_Rec_Fonds[self.Firm_Existe] = self.Fonds_Avoir * self.Firm_Credit_Dem[self.Firm_Existe] / sum(self.Firm_Credit_Dem[self.Firm_Existe])
			self.Firm_Credit_Dem[self.Firm_Existe] -= self.Firm_Credit_Rec_Fonds[self.Firm_Existe]       
		
		###On retire les crédits pris par les entreprises du fonds
		tot_fonds_invest = sum(self.Firm_Credit_Rec_Fonds[self.Firm_Existe])
		self.Fonds_Avoir -= tot_fonds_invest
		self.Fonds_Yield = tot_fonds_invest
		self.Fonds_Avoir = max(self.Fonds_Avoir, 0.0) ##Pour ne pas voir une quantité d'argent négative due aux pbs d'arronds

		###On ajoute les prêts reçus aux avoirs
		self.Firm_Avoir[self.Firm_Existe] += self.Firm_Credit_Rec_Fonds[self.Firm_Existe]
		##Initialisation des coûts
		self.Firm_Cost[self.Firm_Existe] += self.Taux_Interet * self.Firm_Credit_Rec_Fonds[self.Firm_Existe]


	def firm_credit_dem_bank(self, sens_new_info=1.0, prob_transform= lambda x: x) -> None:
		##To select the right firms
		Exist_N_Durable = self.Firm_Existe & np.logical_not(self.Firm_Durable)
		Exist_Durable = self.Firm_Existe & self.Firm_Durable

		### CREDITS BANCAIRES ### ##Le firmes durables reçoivent les crédits voulus
		self.Firm_Credit_Rec_Bank[Exist_Durable] = self.Firm_Credit_Dem[Exist_Durable]
		self.Firm_Credit_Dem[Exist_Durable] = 0.0

		#3.d. Selon la différence entre la proportion effective et le taux exogène imposé, les banques font une offre de crédits aux entreprises non durable selon un processus en « fontaine ». 
		Demande_Credit_N_durable = sum(self.Firm_Credit_Dem[Exist_N_Durable])

		#Crédit total non durable à donner
		if self.Prop_Durable > self.tol:
			Offre_Credit_N_Durable = (1 - self.Prop_Durable) / self.Prop_Durable * sum(self.Firm_Credit_Rec_Bank[Exist_Durable])
		else:
			Offre_Credit_N_Durable = Demande_Credit_N_durable + self.tol


		if Offre_Credit_N_Durable >= Demande_Credit_N_durable:
			self.Firm_Credit_Rec_Bank[Exist_N_Durable] = self.Firm_Credit_Dem[Exist_N_Durable]
			self.Firm_Credit_Dem[Exist_N_Durable] = 0.0
		else:
			self.Firm_Credit_Rec_Bank[Exist_N_Durable] = Offre_Credit_N_Durable * \
														 self.Firm_Credit_Dem[Exist_N_Durable] / Demande_Credit_N_durable
			self.Firm_Credit_Dem[Exist_N_Durable] -= self.Firm_Credit_Rec_Bank[Exist_N_Durable]
		#Recoit est le tableau des indices de ceux qui ont recu moins que ce qu'ils ont demandé (et peuvent donc encore recevoir)
			#Recoit = Firm_Credit_Rec_Bank[Firm_Existe & np.logical_not(Firm_Durable)] < Firm_Credit_Dem[Firm_Existe & np.logical_not(Firm_Durable)]

		###On ajoute les prêts reçus aux avoirs
		self.Firm_Avoir[self.Firm_Existe] += self.Firm_Credit_Rec_Bank[self.Firm_Existe]

		###Update probability of firms to become sustainable
		self.Prob_Become_Durable[Exist_N_Durable] = (1.0 - sens_new_info) * self.Prob_Become_Durable[Exist_N_Durable] + \
								  					sens_new_info * self.Firm_Credit_Dem[Exist_N_Durable] / (self.Firm_Avoir[Exist_N_Durable] + self.Firm_Credit_Dem[Exist_N_Durable])

		#3.f.   Les entreprises évaluent ensuite ces deux offres ensemble et les compare au profit qu’ils feraient s’ils devenaient durables.
		### EQUATION SUR PAPIER MEHDI 13 AVRIL.        
		##partie avant 1+Taux_Interet: Nouveau crédit nécessaire pour produire la projection vu la prime durable
		self.Firm_New_Durable &= False
		
		Firm_Select = self.Firm_Existe & (self.Firm_Credit_Dem > self.tol)
		
		self.Firm_New_Durable[Firm_Select] = self.rng.binomial(1, prob_transform(self.Prob_Become_Durable[Firm_Select])) == 1
		self.Firm_New_Durable[self.Firm_Existe] &= np.logical_not(self.Firm_Durable[self.Firm_Existe])

		###Les firmes nouvellement devenues durables réévaluent leur besoin en financement (car les salaires vont changer)
		###L'addition de (Prime_D * Firm_Avoir) est là pour compenser le fait que Firm_Credit_Dem prend déjà en compte les avoirs
		self.Firm_Credit_Dem[self.Firm_New_Durable] = np.maximum(0.0, self.Firm_Production[self.Firm_New_Durable] * (1.0 + self.Prime_D) - \
													  				  self.Firm_Avoir[self.Firm_New_Durable])
		#Les firmes ayant reçu moins d'argent que demandé deviennent durables, elles reçoivent donc le financement demandé
		self.Firm_Credit_Rec_Bank[self.Firm_New_Durable] += self.Firm_Credit_Dem[self.Firm_New_Durable]
		
		self.Firm_Avoir[self.Firm_New_Durable] += self.Firm_Credit_Dem[self.Firm_New_Durable]
		
		self.Firm_Credit_Dem[self.Firm_New_Durable] = 0.0

		##Calcul du ratio entre fonds obtenus et fonds demandés
		self.Firm_Credit_Miss_Dem_Ratio[self.Firm_Existe] = np.where(self.Firm_Credit_Dem[self.Firm_Existe] > self.tol,
							       									 self.Firm_Credit_Dem[self.Firm_Existe] / \
							      									 (self.Firm_Credit_Dem[self.Firm_Existe] + self.Firm_Credit_Rec_Bank[self.Firm_Existe] + self.Firm_Credit_Rec_Fonds[self.Firm_Existe]),
																	 0.0)

		###Mise à jour des coûts
		self.Firm_Cost[self.Firm_Existe] += self.Taux_Interet * self.Firm_Credit_Rec_Bank[self.Firm_Existe]

		## On change leur indice de durabilité en firmes durables
		self.Firm_Durable[self.Firm_Existe] = self.Firm_Durable[self.Firm_Existe] | self.Firm_New_Durable[self.Firm_Existe]
			

	def firm_produce(self) -> None:
		###Les firmes ayant décidé de rester non durables mettent à jour la quantité qu'elles vont produire
		##Les firmes durables produisent exactement les quantités planifiées
		#select = self.Firm_Existe & np.logical_not(self.Firm_Durable)
		self.Firm_Production[self.Firm_Existe] = np.minimum(self.Firm_Production[self.Firm_Existe],
															self.Firm_Avoir[self.Firm_Existe] / self.Salaire / \
															(1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]))

		self.Firm_Last_Production[self.Firm_Existe] = self.Firm_Production[self.Firm_Existe]

		self.Firm_Cost[self.Firm_Existe] += self.Firm_Production[self.Firm_Existe] * \
											self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe])
												### Pourquoi divisé par salaire, et pas "-" salaire?
		###On ajoute temporairement les prets aux Avoirs des entreprises
		#self.Firm_Avoir[self.Firm_Existe] += self.Firm_Credit_Rec_Bank[self.Firm_Existe] + self.Firm_Credit_Rec_Fonds[self.Firm_Existe] 
		###Adjust production with respect to assets
		#self.Firm_Production[self.Firm_Existe] = np.minimum(self.Firm_Production[self.Firm_Existe],
		#													(self.Firm_Avoir[self.Firm_Existe] self.Firm_Credit_Rec_Bank[self.Firm_Existe] + self.Firm_Credit_Rec_Fonds[self.Firm_Existe])) / \
		#													self.Salaire / (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]))

	def firm_set_price(self) -> None:
		self.Firm_Prix[self.Firm_Existe] = np.where(self.Firm_Production[self.Firm_Existe] < self.tol,
													1.0, ##Afin d'éviter une division par zero, dans la ligne suivante on met une condition qui n'a pas d'importance
													(1 + self.Firm_Markup) / np.where(self.Firm_Production[self.Firm_Existe] < self.tol, 1.0, self.Firm_Production[self.Firm_Existe]) * \
													self.Firm_Cost[self.Firm_Existe])
													##Dépenses
													#(self.Salaire / self.Firm_Productivite[self.Firm_Existe] * self.Firm_Production[self.Firm_Existe] + \
	      											# self.Taux_Interet * (self.Firm_Credit_Rec_Bank[self.Firm_Existe] + self.Firm_Credit_Rec_Fonds[self.Firm_Existe]) + \
													# self.Firm_Fixed_Cost))
		
	def consume(self) -> None:
		#5.b.Les consommateurs choisissent les produits au hasard et en fonction des prix.
		####Only one round of purchase ?? we can make it with several rounds until income or all firm stocks are depleted
		###WE DO SEVERAL CONSUMPTION ROUNDS ---> TO DO

		firm_selling = self.Firm_Existe & (self.Firm_Production > self.tol)

		self.Firm_Demande[firm_selling] = self.Avail_Wealth * self.Firm_Prix[firm_selling]**(-(self.Sens_Prix + 1.0)) / \
										  sum(self.Firm_Prix[firm_selling]**(-self.Sens_Prix))

		
		####On augmente les avoirs de la firme par ses revenus de la vente (et non des dividendes)
		self.Firm_Avoir[firm_selling] += self.Firm_Prix[firm_selling] * np.minimum(self.Firm_Demande[firm_selling], self.Firm_Production[firm_selling])

		self.Last_Total_Consumption = sum(self.Firm_Prix[firm_selling] * np.minimum(self.Firm_Demande[firm_selling], self.Firm_Production[firm_selling]))

		self.Avail_Wealth -= self.Last_Total_Consumption

	def consume_max_income(self) -> None:
		#5.b.Les consommateurs choisissent les produits au hasard et en fonction des prix.
		####Only one round of purchase ?? we can make it with several rounds until income or all firm stocks are depleted
		###WE DO SEVERAL CONSUMPTION ROUNDS ---> TO DO
		self.Last_Total_Consumption = 0.0
		##select selling firms
		firm_selling = self.Firm_Existe & (self.Firm_Production > self.tol)
		
		while np.any(firm_selling) & (self.Avail_Wealth > self.tol):
			self.Firm_Demande[firm_selling] = self.Avail_Wealth * self.Firm_Prix[firm_selling]**(-(self.Sens_Prix + 1.0)) / \
											  sum(self.Firm_Prix[firm_selling]**(-self.Sens_Prix))

			###Update income and total consumption
			spent_cons = sum(self.Firm_Prix[firm_selling] * np.minimum(self.Firm_Demande[firm_selling], self.Firm_Production[firm_selling]))
			
			self.Avail_Wealth -= spent_cons
			self.Last_Total_Consumption += spent_cons

			###Update firms stocks, assets and selling firms
			self.Firm_Avoir[firm_selling] += self.Firm_Prix[firm_selling] * np.minimum(self.Firm_Demande[firm_selling], self.Firm_Production[firm_selling])

			self.Firm_Production[firm_selling] -= np.minimum(self.Firm_Demande[firm_selling], self.Firm_Production[firm_selling])

			firm_selling[firm_selling] &= self.Firm_Production[firm_selling] > self.tol
	

	def firm_pay_bank(self) -> None:
		###Ici, il ne s'agit pas de Crédit reçu à proprement parler, mais on stocke les créances vis à vis
		###de la banque dans ce vecteur afin de ne pas en créer de nouveau (performance du code)
		
		###Remboursement de la banque
		###Intérêt des prêts reversés aux consommateurs
		self.Avail_Wealth += sum(np.minimum(self.Firm_Credit_Rec_Bank[self.Firm_Existe] * self.Taux_Interet,
											np.maximum(self.Firm_Avoir[self.Firm_Existe] - self.Firm_Credit_Rec_Bank[self.Firm_Existe], 0.0)))

		#np.minimum(self.Firm_Credit_Rec_Bank[self.Firm_Existe], np.maximum(self.Firm_Avoir[self.Firm_Existe], 0.0)))
		
		self.Firm_Avoir[self.Firm_Existe] -= (1.0+self.Taux_Interet) * self.Firm_Credit_Rec_Bank[self.Firm_Existe]

	###Remboursement du fonds
	def firm_pay_fund(self) -> None:
		tot_fonds_payment = sum(np.minimum((1.0 + self.Taux_Interet) * self.Firm_Credit_Rec_Fonds[self.Firm_Existe],
									   	   np.maximum(self.Firm_Avoir[self.Firm_Existe], 0.0)))

		self.Fonds_Avoir += tot_fonds_payment
		self.Fonds_Yield = 0.0 if self.Fonds_Yield == 0.0 else (tot_fonds_payment - self.Fonds_Yield) / self.Fonds_Yield

		self.Firm_Avoir[self.Firm_Existe] -= (1.0+self.Taux_Interet) * self.Firm_Credit_Rec_Fonds[self.Firm_Existe]


	def cons_withdraw_fund(self) -> None:
		withdrawn = (1.0 - (self.Fonds_Yield + 1.0) / (2.0 + self.Taux_Interet)) * self.Fonds_Avoir
		###Les consommateurs retirent du fonds
		self.Avail_Wealth += withdrawn
		self.Fonds_Avoir -= withdrawn


	def firm_pay_salary(self) -> None:
		tot_salary = sum(np.minimum(self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] + \
									  self.Firm_Fixed_Cost,
									  np.maximum(self.Firm_Avoir[self.Firm_Existe], 
		      									 0.0)))

		##Paiement des salaires
		self.Avail_Wealth += tot_salary
		self.Income += tot_salary
		self.Firm_Avoir[self.Firm_Existe] -= self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] + \
											 self.Firm_Fixed_Cost
		

	def firm_exit(self) -> None:
		###Firmes qui ont un avoir négatif quittent le marché
		self.Firm_Existe[self.Firm_Existe] &= (self.Firm_Avoir[self.Firm_Existe] >= 0.0)

	##Firms decide on their assets depending on next funds
	def firm_budgetize_and_pay_shareholders(self, prop_cost_keep=0.0) -> None:
		###Total payments to shareholders
		tot_sh_pay = sum(np.maximum(self.Firm_Avoir[self.Firm_Existe] - prop_cost_keep * self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] - \
									  self.Firm_Fixed_Cost,
									  0.0))
		##Pay shareholders
		self.Avail_Wealth += tot_sh_pay
		self.Income += tot_sh_pay
		## Update firms assets
		self.Firm_Avoir[self.Firm_Existe] = np.minimum(prop_cost_keep * self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] + \
									  				   self.Firm_Fixed_Cost,
						 							   self.Firm_Avoir[self.Firm_Existe])
		
	def firm_budgetize_and_pay_shareholders_dynamic(self, min_prop_cost_keep) -> None:
		tot_sh_pay = sum(np.maximum(self.Firm_Avoir[self.Firm_Existe] - 
			      					np.maximum(self.Firm_Credit_Miss_Dem_Ratio[self.Firm_Existe], min_prop_cost_keep) * \
						 			self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] - \
						 			self.Firm_Fixed_Cost,
									0.0))
		##Pay shareholders
		self.Avail_Wealth += tot_sh_pay
		self.Income += tot_sh_pay
		
		self.Firm_Avoir[self.Firm_Existe] = np.minimum(np.maximum(self.Firm_Credit_Miss_Dem_Ratio[self.Firm_Existe], min_prop_cost_keep) * \
						 							   self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] - \
						 							   self.Firm_Fixed_Cost,
													   self.Firm_Avoir[self.Firm_Existe])



	def firm_pay_shareholders(self, prop_cost_keep=0.0) -> None:
		tot_sh_pay = sum(np.maximum(self.Firm_Avoir[self.Firm_Existe] - prop_cost_keep * self.Firm_Cost[self.Firm_Existe], 0.0))
		self.Income += tot_sh_pay
		self.Avail_Wealth += tot_sh_pay
		self.Firm_Avoir[self.Firm_Existe] = np.minimum(self.Firm_Avoir[self.Firm_Existe],
						 							   prop_cost_keep * self.Firm_Cost[self.Firm_Existe])
		
	def cons_invest_fund(self) -> None:
		###Les consommateurs investissent dans le fonds
		curr_invest_fonds = self.Taux_Invest_Fonds * self.Income
		self.Fonds_Avoir += curr_invest_fonds
		self.Avail_Wealth -= curr_invest_fonds

	def increment_iteration(self) -> None:
		self.Iteration += 1

	####Economy information methods
	def nb_firms(self):
		return sum(self.Firm_Existe)

	def nb_sustainable_firms(self):
		return sum(self.Firm_Durable[self.Firm_Existe])

	def gross_product(self):
		return self.Last_Total_Consumption
	
	def sustainable_consumed_quantity(self):
		return sum(self.Firm_Last_Production[self.Firm_Existe & self.Firm_Durable] - self.Firm_Production[self.Firm_Existe & self.Firm_Durable])

	def non_sustainable_consumed_quantity(self):
		return sum(self.Firm_Last_Production[self.Firm_Existe & np.logical_not(self.Firm_Durable)] - self.Firm_Production[self.Firm_Existe & np.logical_not(self.Firm_Durable)])

	def total_money(self):
		return sum(self.Firm_Avoir[self.Firm_Existe]) + self.Avail_Wealth + self.Fonds_Avoir

	def total_credit(self):
		return sum(self.Firm_Credit_Rec_Bank[self.Firm_Existe] + self.Firm_Credit_Rec_Fonds[self.Firm_Existe])

	def fund_assets(self):
		return self.Fonds_Avoir

T_Max = 1400

t_invest_val = [0.25]

n_dur_fin = np.zeros(len(t_invest_val))
n_ndur_fin = np.zeros(len(t_invest_val))

q_dur_fin = np.zeros(len(t_invest_val))
q_ndur_fin = np.zeros(len(t_invest_val))

for tif in range(len(t_invest_val)):

	eco = Economy(Taux_Interet= 0.05, Firm_Fixed_Cost= 0.0, Prime_D = 0.2, Firm_Markup = 0.1, Prod_Ajust = 0.2, 
				 Taux_Invest_Fonds= t_invest_val[tif], Init_Wealth= 0.0, Sens_Prix = 2.5, N_Firm = T_Max)

	gdp = np.zeros(T_Max)
	total_money = np.zeros(T_Max)
	total_firm_avoir = np.zeros(T_Max)
	n_firms_durable = np.zeros(T_Max)
	n_firms = np.zeros(T_Max)
	fonds_avoir = np.zeros(T_Max)

	sus_consumed_quantity = np.zeros(T_Max)
	n_sus_consumed_quantity = np.zeros(T_Max)

	#t0 = time.time()

	gc.collect(generation=2)
	gc.collect(generation=1)

	t_min_durable = 1500
	for t in range(T_Max):
		if t % 200 == 0: print("Step:", t, ";")
		if ((t/T_Max) > t_min_durable) & (t % 200 == 0):
			print("Prop durable change to : ", min(9.0 * (t / T_Max - t_min_durable), 1.0))
			eco.set_prop_durable(min(9.0 * (t / T_Max - t_min_durable), 1.0))
		eco.firm_enter(10, prob_durable=0.0)
		eco.firm_decide_quantity()
		eco.reinitialize_income()
		#eco.firm_budgetize_and_pay_shareholders(prop_cost_keep=0.1) ##income ++
		eco.firm_budgetize_and_pay_shareholders_dynamic(min_prop_cost_keep=0.5) ##income ++
		#print(eco.Firm_Production[eco.Firm_Existe])
		#time.sleep(0.1)
		#print(eco.Firm_Production[eco.Firm_Existe])
		#print("---------Step ", t, " ----------------")
		eco.firm_decide_credit()
		eco.firm_credit_dem_fund()
		eco.firm_credit_dem_bank(sens_new_info= 0.01, prob_transform= lambda x : x**1.0)
		#print("Credit rec.:", eco.Firm_Credit_Rec_Bank[eco.Firm_Existe])
		#time.sleep(0.5)
		#input()
		eco.firm_produce()
		#print("Firm Rec Fonds: ", eco.Firm_Credit_Rec_Fonds[eco.Firm_Existe])
		eco.firm_set_price()
		#print("Avoir.:",eco.Firm_Avoir[eco.Firm_Existe])
		#input()
		eco.firm_pay_salary() ##income ++
		eco.cons_invest_fund() ## wealth --
		eco.consume_max_income() ## wealth --
		eco.firm_pay_fund() ## fonds ++
		eco.firm_pay_bank() ## wealth ++, but not income
		
		eco.cons_withdraw_fund()
		
		
		eco.firm_exit()

		eco.increment_iteration()
		#input()
		#print(eco.Firm_Avoir)
		#print(eco.Firm_Avoir)
		#print(eco.Firm_Avoir) 
		#print(eco.Firm_Avoir)
		gdp[t] = eco.gross_product()
		total_money[t] = eco.total_money()
		total_firm_avoir[t] = np.sum(eco.Firm_Avoir[eco.Firm_Existe])
		n_firms[t] = eco.nb_firms()
		n_firms_durable[t] = eco.nb_sustainable_firms()
		sus_consumed_quantity[t] = eco.sustainable_consumed_quantity()
		n_sus_consumed_quantity[t] = eco.non_sustainable_consumed_quantity()
		fonds_avoir[t] = eco.Fonds_Avoir

	n_dur_fin[tif] = eco.nb_sustainable_firms()
	n_ndur_fin[tif] = eco.nb_firms() - n_dur_fin[tif]
	q_dur_fin[tif] = eco.sustainable_consumed_quantity()
	q_ndur_fin[tif] = eco.non_sustainable_consumed_quantity()



plt.plot(t_invest_val, n_ndur_fin, c= 'red'); plt.plot(t_invest_val, n_dur_fin, c = 'green'); plt.show()
plt.plot(t_invest_val, q_ndur_fin, c= 'red'); plt.plot(t_invest_val, q_dur_fin, c = 'green'); plt.show()


###########################################
#GARDER LES AVOIRS EN FONCTION DE LA PREVISION DE PRODUCTION FUTURE
#IL EST POSSIBLE QUE L'ORDRE D'APPEL DES FONCTIONS CHANGE
###########################################


#eco.Firm_Production[eco.Firm_Existe]
#eco.Firm_Productivite[eco.Firm_Existe]
#eco.Firm_Avoir[eco.Firm_Existe]

#t1 = time.time()
#t1 - t0

##ajouter des épargnants qui veulent maximiser leur rémunération par leur épargne, épargne investie
plt.semilogy(gdp); plt.show()

#plt.hist(eco.Firm_Avoir[eco.Firm_Existe])

plt.plot(total_money); plt.plot(total_firm_avoir); plt.show()

plt.plot(total_firm_avoir / total_money, c = 'orange'); plt.plot(fonds_avoir / total_money, c = 'purple'); plt.show()

plt.plot(n_firms - n_firms_durable, c= 'red'); plt.plot(n_firms_durable, c = 'green'); plt.show()

plt.plot(n_sus_consumed_quantity, c = 'red'); plt.plot(sus_consumed_quantity, c = 'green'); plt.show()

plt.plot(sus_consumed_quantity / (n_sus_consumed_quantity + sus_consumed_quantity), c = 'green'); plt.show()

plt.scatter(eco.Firm_Moment_Creation[eco.Firm_Existe], eco.Firm_Avoir[eco.Firm_Existe], c = np.where(eco.Firm_Durable[eco.Firm_Existe], 'green', 'red')); plt.show()

plt.scatter(eco.Firm_Moment_Creation[eco.Firm_Existe], eco.Firm_Last_Production[eco.Firm_Existe], c = np.where(eco.Firm_Durable[eco.Firm_Existe], 'green', 'red')); plt.show()

plt.scatter(eco.Firm_Moment_Creation[eco.Firm_Existe], eco.Firm_Prix[eco.Firm_Existe], c = np.where(eco.Firm_Durable[eco.Firm_Existe], 'green', 'red')); plt.show()

plt.scatter(eco.Firm_Moment_Creation[eco.Firm_Existe], eco.Firm_Credit_Miss_Dem_Ratio[eco.Firm_Existe], c = np.where(eco.Firm_Durable[eco.Firm_Existe], 'green', 'red'))

plt.plot(fonds_avoir); plt.show()

plt.loglog(np.sort(eco.Firm_Last_Production[eco.Firm_Existe])[::-1])

eco.Income

eco.Fonds_Avoir

sum(eco.Firm_Avoir[eco.Firm_Existe])

#plt.plot(eco.Firm_Production[eco.Firm_Existe]); plt.show()



m = (np.sin(np.arange(0., 100., 0.3)) + 1)
p = np.ones(len(m)) * m[0]

th = 0.96

for i in range(1, len(m)):
	p[i] = th * p[i-1] + (1. - th) * m[i]

plt.plot(p); plt.plot(m);

