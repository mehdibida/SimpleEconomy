# Modèle : est-ce que les banques ont une influence sur la société 
import numpy as np

###Checker si les crédits doivent ou pas aller dans les avoirs des firmes et le faire le cas échéant.

class Economy:
	def __init__(self, 
			  	 Max_Resource, Res_Recovery_Rate, 
			  	 Taux_Entrep, Taux_Interet, 
				 Firm_Fixed_Cost, Prime_D, Firm_Markup, Labor_Rigidity,
			  	 Prod_Adjust,
				 Population, Contract_Duration,
				 Taux_Invest_Fonds, Init_Wealth, 
				 Sens_Prix, Init_Prop_Durable=0.0, N_Firm = 100, tol = 10.0**-6) -> None:
		##resource levels
		self.Resource = Max_Resource
		self.Res_Recovery_Rate = Res_Recovery_Rate
		self.Max_Resource = Max_Resource
		self.Resource_Threshold = 0.0
		


		#self.Prop_Durable = Init_Prop_Durable  #Proportion minimum de prêt durable, initialement à zero
		self.Taux_Interet = Taux_Interet #Le taux d'intérêt des Fonds_Avoirs/crédits
		
		self.Income = 0.0 ##Income received during iteration

		self.Last_Total_Consumption = 0.0

		self.Prev_Last_Total_Prod = 0.0 #Total production at step t-2
		self.Last_Total_Prod = 0.0 #Total production at step t-1
		
		self.Contract_Duration = Contract_Duration #Number of steps until a hired worker is released
		self.Population = Population #Total population

		#self.Qualité_Durable = 2.0 #Qualité des produits vendus par les firmes durables, qualité des non-durables à 1
		#Salaires
		self.Taux_Entrep = Taux_Entrep #Taux d'entreprenariat: combien on prélève pour la création de nouvelles entreprises
		self.Prime_D = Prime_D #Prime pour les salaires durables (facteur) #Il est en hashtag dans le modèle V1 : pourquoi ?
		self.Salaire = 1.0 #Salaire non durable
		self.Firm_Fixed_Cost = Firm_Fixed_Cost
		self.Avail_Wealth = Init_Wealth
		self.Firm_Markup = Firm_Markup
		self.Labor_Rigidity = Labor_Rigidity
		self.Taux_Invest_Fonds = Taux_Invest_Fonds
		self.Taux_Retrait_Fonds = 0.0
		self.Sens_Prix = Sens_Prix
		self.Prod_Adjust = Prod_Adjust
		
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
		self.Firm_Labor_Last = np.zeros(N_Firm) #Firm labor need
		self.Firm_Labor = np.zeros(N_Firm) #Nombre de travailleurs pour chaque itération dans la durée du contrat
		self.Firm_Labor_Hiring = np.zeros(N_Firm) # amount of labor that the firm needs to recruit
		self.Firm_Cost = np.zeros(N_Firm) #Niveau des profits faits par l'entreprise
		self.Firm_Avoir = np.zeros(N_Firm) #Avoir de l'entreprise
		self.Firm_Credit_Score = np.zeros(N_Firm) ##Indique aux créanciers la qualité de l'entreprise en tant que débiteur
		self.Firm_Credit_Dem = np.zeros(N_Firm) #Credits demandés
		self.Firm_Credit_Rec_Bank = np.zeros(N_Firm) #Credits reçu par l'entreprise (octroyé par la banque)
		self.Firm_Credit_Rec_Fonds = np.zeros(N_Firm)
		self.Firm_Credit_Rec_Last = np.zeros(N_Firm) #Ratio entre les fonds reçus et les fonds demandés
		self.Firm_Demande = np.zeros(N_Firm) #Quantités demandées à chacune de entreprises
		self.Firm_Prix = np.ones(N_Firm) #Prix des entreprises

		self.Prob_Become_Durable = np.zeros(N_Firm)		

		##Caractéristiques des fonds
		self.Fonds_Avoir = 0.0  #Taille du fonds (initialization à zero)
		self.Fonds_Yield = 0.0

		###Random number generator
		self.rng = np.random.default_rng()

	#def set_prop_durable(self, prop) -> None:
	#	self.Prop_Durable = prop

	def set_res_threshold(self, threshold) -> None:
		self.Resource_Threshold = threshold

	def reinitialize_income(self) -> None:
		self.Income = 0.0

	def firm_enter(self, prob_durable=0.0, Capital_Dist=np.random.exponential, **dist_kwargs):
		##Capital available to create firms
		n_entrepreneurs = self.Taux_Entrep * self.Population
		### Keep average available wealth for random generation of capital
		avg_avail_wealth = self.Avail_Wealth / self.Population
		##Labor of new firms
		new_firm_labor = np.zeros(np.ceil(n_entrepreneurs * 1.5, dtype = np.int32))
		##Capital of entering firms
		new_firm_cap = np.zeros(np.ceil(n_entrepreneurs * 1.5, dtype = np.int32))
		n_created = 0
		while (n_entrepreneurs > self.tol) & (self.Avail_Wealth > self.tol):
			## Extend array if we still need to create firms
			if n_created == len(new_firm_labor):
				new_firm_labor.resize(new_firm_labor, np.ceil(n_created * 1.5, dtype = np.uint32), refcheck = False)
				new_firm_cap.resize(new_firm_cap, np.ceil(n_created * 1.5, dtype = np.uint32), refcheck = False)
			## Decide on new firms labor and financial capital, idea:
			## firm labor depends exponentially on the number of people wanting to associate: each person wants to associate with a certain prob, they all need to agree to be in the same firm
			## firm capital is the sum of wealth of associates: assuming wealth of population has exponential distribution
			new_firm_labor[n_created] = np.min(np.random.exponential(1.0), n_entrepreneurs)
			new_firm_cap[n_created] = np.min(np.random.gamma(new_firm_labor[n_created], avg_avail_wealth), self.Avail_Wealth)
			### Subtract entrepreneurs and capital from available
			n_entrepreneurs -= new_firm_labor[n_created]
			self.Avail_Wealth -= new_firm_cap[n_created]
			n_created += 1

		##Extend vectors if there is no sufficient place for all active firms
		if len(new_firm_cap) > (len(self.Firm_Existe) - sum(self.Firm_Existe)):
			extension_size = len(new_firm_labor) - sum(np.logical_not(self.Firm_Existe))
			self.Firm_Existe.resize(len(self.Firm_Existe) + extension_size, refcheck = False)
			self.Firm_Durable.resize(len(self.Firm_Durable) + extension_size, refcheck = False)
			self.Firm_Last_Production.resize(len(self.Firm_Last_Production) + extension_size, refcheck = False)
			self.Firm_New_Durable.resize(len(self.Firm_New_Durable) + extension_size, refcheck = False)
			self.Firm_Production.resize(len(self.Firm_Production) + extension_size, refcheck = False)
			self.Firm_Labor.resize(len(self.Firm_Labor) + extension_size, refcheck = False)
			self.Firm_Labor_Last.resize(len(self.Firm_Labor_Last) + extension_size, refcheck = False)
			self.Firm_Labor_Hiring.resize(len(self.Firm_Labor_Hiring) + extension_size, refcheck = False)
			self.Firm_Cost.resize(len(self.Firm_Cost) + extension_size, refcheck = False)
			self.Firm_Avoir.resize(len(self.Firm_Avoir) + extension_size, refcheck = False)
			self.Firm_Credit_Score.resize(len(self.Firm_Credit_Score) + extension_size, refcheck = False)
			self.Firm_Credit_Dem.resize(len(self.Firm_Credit_Dem) + extension_size, refcheck = False)
			self.Firm_Credit_Rec_Bank.resize(len(self.Firm_Credit_Rec_Bank) + extension_size, refcheck = False)
			self.Firm_Credit_Rec_Fonds.resize(len(self.Firm_Credit_Rec_Fonds) + extension_size, refcheck = False)
			self.Firm_Credit_Rec_Last.resize(len(self.Firm_Credit_Rec_Last) + extension_size, refcheck = False)
			self.Firm_Demande.resize(len(self.Firm_Demande) + extension_size, refcheck = False)
			self.Firm_Prix.resize(len(self.Firm_Prix) + extension_size, refcheck = False)
			self.Firm_Moment_Creation.resize(len(self.Firm_Moment_Creation) + extension_size, refcheck = False)
			self.Prob_Become_Durable.resize(len(self.Prob_Become_Durable) + extension_size, refcheck = False)

			self.Firm_Labor = np.append(self.Firm_Labor, np.zeros((extension_size, self.Contract_Duration)))

			self.Firm_Prix[np.logical_not(self.Firm_Existe)] = 1.0
		###Initialize new firms
		#Liste d'indices dispos qui correspondent aux cases qui vont contenir les nouvelles entreprises
		Indices_Entrants = np.flatnonzero(np.logical_not(self.Firm_Existe))[:len(new_firm_cap)] ##Trouver les indices qui sont False
		self.Firm_Last_Production[Indices_Entrants] = 0.0
		self.Firm_Production[Indices_Entrants] = 0.0
		self.Firm_Cost[Indices_Entrants] = 0.0
		self.Firm_Avoir[Indices_Entrants] = new_firm_cap
		self.Firm_Credit_Score[Indices_Entrants] = 0.0
		self.Firm_Credit_Dem[Indices_Entrants] = 0.0
		self.Firm_Credit_Rec_Bank[Indices_Entrants] = 0.0
		self.Firm_Credit_Rec_Fonds[Indices_Entrants] = 0.0
		self.Firm_Durable[Indices_Entrants] = self.rng.binomial(1, prob_durable, len(new_firm_cap)) == 1
		self.Firm_Existe[Indices_Entrants] = True
		self.Firm_Moment_Creation[Indices_Entrants] = self.Iteration
		self.Prob_Become_Durable[Indices_Entrants] = 0.0
		return len(new_firm_cap)

	def firm_decide_quantity(self):
		new_entrants = np.flatnonzero(self.Firm_Existe & (self.Firm_Moment_Creation == self.Iteration))
		incumbents = np.flatnonzero(self.Firm_Existe & (self.Firm_Moment_Creation < self.Iteration))
		
		if len(incumbents) > 0:
			##2.b : Les entreprises existantes ajustent la production par rapport aux ventes précédentes. Prod_Sensibilite entre 0 et 1 et correspond à la sensibilité des entrepeneurs.
			self.Firm_Production[incumbents] =  (self.Firm_Last_Production[incumbents] - self.Firm_Production[incumbents]) * \
										   		(1.0 + self.Prod_Adjust * np.random.uniform(0.0, 1.0, len(incumbents)))
			
		if len(new_entrants) > 0:
			self.Firm_Production[new_entrants] = np.maximum((self.Firm_Avoir[new_entrants] - self.Firm_Fixed_Cost) / \
					  							   		 	self.Salaire / (1.0 + self.Prime_D * self.Firm_Durable[new_entrants]) * \
												 		 	np.random.uniform(0.75, 1.25, size = len(new_entrants)),
														 	self.Firm_Labor[new_entrants])

		##Initialisation des coûts aux coûts fixes
		self.Firm_Cost[self.Firm_Existe] = self.Firm_Fixed_Cost

		return self.Firm_Production[self.Firm_Existe]

	##Firms decide on their assets depending on next funds
	def firm_budgetize_and_pay_shareholders_static(self, prop_cost_keep=0.0):
		###Total payments to shareholders
		tot_sh_pay = np.sum(np.maximum(self.Firm_Avoir[self.Firm_Existe] - prop_cost_keep * self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] - \
									   self.Firm_Fixed_Cost,
									   0.0))
		##Pay shareholders
		self.Avail_Wealth += tot_sh_pay
		self.Income += tot_sh_pay
		## Update firms assets
		self.Firm_Avoir[self.Firm_Existe] = np.minimum(prop_cost_keep * self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] + \
									  				   self.Firm_Fixed_Cost,
						 							   self.Firm_Avoir[self.Firm_Existe])
		
		return self.Firm_Avoir[self.Firm_Existe]
	
	##Firms decide on their assets depending on next funds and credit received relatively to demanded
	def firm_budgetize_and_pay_shareholders_dynamic(self, min_prop_cost_keep):
		tot_sh_pay = np.sum(np.maximum(self.Firm_Avoir[self.Firm_Existe] - 
			      					   np.maximum(self.Firm_Credit_Rec_Last[self.Firm_Existe], min_prop_cost_keep) * \
						 			   self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] - \
						 			   self.Firm_Fixed_Cost,
									   0.0))
		##Pay shareholders
		self.Avail_Wealth += tot_sh_pay
		self.Income += tot_sh_pay
		## Update firms assets
		firm_op_cost = self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] + \
					   self.Firm_Fixed_Cost
		self.Firm_Avoir[self.Firm_Existe] = np.maximum(firm_op_cost - self.Firm_Credit_Rec_Last[self.Firm_Existe],
													   firm_op_cost * min_prop_cost_keep)
		
		return self.Firm_Avoir[self.Firm_Existe]

	def firm_decide_credit(self):
		##Firms decisde on how much credit they need
		self.Firm_Credit_Dem[self.Firm_Existe] = np.maximum(self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * \
						      								self.Firm_Production[self.Firm_Existe] + \
														    self.Firm_Fixed_Cost - \
															self.Firm_Avoir[self.Firm_Existe], 
															0.0)
		##Credit score given needed quantity of credit
		exist_debitor = self.Firm_Existe & (self.Firm_Credit_Dem >= self.tol)
		self.Firm_Credit_Score[exist_debitor] = self.Firm_Avoir[exist_debitor] / self.Firm_Credit_Dem[exist_debitor]
		self.Firm_Credit_Score[self.Firm_Existe & (self.Firm_Credit_Dem < self.tol)] = np.Inf
		##Reinitialize credit
		self.Firm_Credit_Rec_Fonds[self.Firm_Existe] = 0.0
		self.Firm_Credit_Rec_Bank[self.Firm_Existe] = 0.0

		return self.Firm_Credit_Dem[self.Firm_Existe]

	###Generic function for distrbuting funds, works for bank (from_fund = False) and fund (from_fund = True)
	def distribute_credit(self, avail_credit, from_fund, cs_importance=1.0) -> None:
		if avail_credit >= self.tol: #only enter loop if there is credit available
			##Which firms are borrowing
			firm_borrowing = self.Firm_Existe & (self.Firm_Credit_Dem >= self.tol)
			##Choose the right credit vector
			if from_fund:
				firm_credit_rec = self.Firm_Credit_Rec_Fonds
			else:
				firm_credit_rec = self.Firm_Credit_Rec_Bank
				##Only non sustainable firms are borrowing from bank if funding is limited
				firm_borrowing &= np.logical_not(self.Firm_Durable)

			###Credit distribution:
			while (avail_credit >= self.tol) & np.any(firm_borrowing):
				###Distribute credit according to credit score
				credit_given = np.minimum(avail_credit * self.Firm_Credit_Score[firm_borrowing]**cs_importance / \
										  np.sum(self.Firm_Credit_Score[firm_borrowing]**cs_importance),
										  self.Firm_Credit_Dem[firm_borrowing])
				firm_credit_rec[firm_borrowing] += credit_given
				self.Firm_Credit_Dem[firm_borrowing] -= credit_given

				##Update quantity of available cerdit
				avail_credit -= np.sum(credit_given)

				if not from_fund:
					print('Remaining avail credit: ', avail_credit)

				firm_borrowing[firm_borrowing] &= (self.Firm_Credit_Dem[firm_borrowing] >= self.tol)
			
			##Update fund assets
			if from_fund:
				self.Fonds_Avoir = max(avail_credit, 0.0)

	def firm_credit_dem_fund(self, cs_importance=1.0):
		### Initialisation
		#Fonds  : "invetsissement" correspond à la qté totale de fonds disponible dans le fonds.
		if self.Fonds_Avoir >= sum(self.Firm_Credit_Dem[self.Firm_Existe]):
			self.Firm_Credit_Rec_Fonds[self.Firm_Existe] = self.Firm_Credit_Dem[self.Firm_Existe]
			self.Firm_Credit_Dem[self.Firm_Existe] = 0.0
		else: 
			self.distribute_credit(self.Fonds_Avoir, True, cs_importance)
		
		###On retire les crédits pris par les entreprises du fonds
		tot_fonds_invest = sum(self.Firm_Credit_Rec_Fonds[self.Firm_Existe])
		self.Fonds_Avoir -= tot_fonds_invest
		self.Fonds_Yield = tot_fonds_invest
		self.Fonds_Avoir = max(self.Fonds_Avoir, 0.0) ##Pour ne pas voir une quantité d'argent négative due aux pbs d'arronds

		###On ajoute les prêts reçus aux avoirs
		self.Firm_Avoir[self.Firm_Existe] += self.Firm_Credit_Rec_Fonds[self.Firm_Existe]
		##Initialisation des coûts
		self.Firm_Cost[self.Firm_Existe] += self.Taux_Interet * self.Firm_Credit_Rec_Fonds[self.Firm_Existe]

		return self.Firm_Credit_Rec_Fonds[self.Firm_Existe]

	def firm_credit_dem_bank(self, cs_importance= 1.0, sens_new_info=1.0, prob_transform= lambda x: x):
		##To select the right firms
		Exist_N_Durable = self.Firm_Existe & np.logical_not(self.Firm_Durable)
		Exist_Durable = self.Firm_Existe & self.Firm_Durable

		### CREDITS BANCAIRES ### ##Le firmes durables reçoivent les crédits voulus
		self.Firm_Credit_Rec_Bank[Exist_Durable] = self.Firm_Credit_Dem[Exist_Durable]
		self.Firm_Credit_Dem[Exist_Durable] = 0.0

		#3.d. Selon la différence entre la proportion effective et le taux exogène imposé, les banques font une offre de crédits aux entreprises non durable selon un processus en « fontaine ». 
		Demande_Credit_N_durable = sum(self.Firm_Credit_Dem[Exist_N_Durable])

		#Crédit total non durable à donner
		#if self.Prop_Durable > self.tol:
		#	Offre_Credit_N_Durable = (1.0 - self.Prop_Durable) / self.Prop_Durable * sum(self.Firm_Credit_Rec_Bank[Exist_Durable])
		#else:
		#	Offre_Credit_N_Durable = (Demande_Credit_N_durable + 1.0) * 1.1

		if self.Resource_Threshold > self.tol:
			Offre_Credit_N_Durable = max(self.Resource - self.Resource_Threshold, 0.0)
		else:
			Offre_Credit_N_Durable = (Demande_Credit_N_durable + 1.0) * 1.1

		print('Demande crédit non durable: ', Demande_Credit_N_durable)
		print('Offre crédit non durable: ', Offre_Credit_N_Durable)

		if Offre_Credit_N_Durable >= Demande_Credit_N_durable:
			self.Firm_Credit_Rec_Bank[Exist_N_Durable] = self.Firm_Credit_Dem[Exist_N_Durable]
			self.Firm_Credit_Dem[Exist_N_Durable] = 0.0
		else:
			self.distribute_credit(Offre_Credit_N_Durable, False, cs_importance)
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
				
		self.Firm_New_Durable[Exist_N_Durable] = self.rng.binomial(1, prob_transform(self.Prob_Become_Durable[Exist_N_Durable])) == 1

		###Les firmes nouvellement devenues durables réévaluent leur besoin en financement (car les salaires vont changer)
		###L'addition de (Prime_D * Firm_Avoir) est là pour compenser le fait que Firm_Credit_Dem prend déjà en compte les avoirs
		self.Firm_Credit_Dem[self.Firm_New_Durable] = np.maximum(0.0, self.Firm_Production[self.Firm_New_Durable] * (1.0 + self.Prime_D) + \
							   										  self.Firm_Fixed_Cost - \
													  				  self.Firm_Avoir[self.Firm_New_Durable])
		#Les firmes ayant reçu moins d'argent que demandé deviennent durables, elles reçoivent donc le financement demandé
		self.Firm_Credit_Rec_Bank[self.Firm_New_Durable] += self.Firm_Credit_Dem[self.Firm_New_Durable]
		
		self.Firm_Avoir[self.Firm_New_Durable] += self.Firm_Credit_Dem[self.Firm_New_Durable]

		self.Firm_Credit_Dem[self.Firm_New_Durable] = 0.0

		##Calcul du ratio entre fonds obtenus et fonds demandés
		firm_cmdr = self.Firm_Existe & (self.Firm_Credit_Score < np.Inf) & np.logical_not(self.Firm_Durable)
		self.Firm_Credit_Rec_Last[self.Firm_Existe] = self.Firm_Credit_Rec_Bank[firm_cmdr] + self.Firm_Credit_Rec_Fonds[firm_cmdr]

		###Mise à jour des coûts
		self.Firm_Cost[self.Firm_Existe] += self.Taux_Interet * self.Firm_Credit_Rec_Bank[self.Firm_Existe]

		## On change leur indice de durabilité en firmes durables
		self.Firm_Durable[self.Firm_Existe] = self.Firm_Durable[self.Firm_Existe] | self.Firm_New_Durable[self.Firm_Existe]

		return self.Firm_Credit_Rec_Bank[self.Firm_Existe]

	##Apply financial constraint: cannot produce more than what funds allow
	def firm_adjust_production_to_equity(self):
		self.Firm_Production[self.Firm_Existe] = np.minimum(self.Firm_Production[self.Firm_Existe],
															np.maximum(0.0,
			  														   (self.Firm_Avoir[self.Firm_Existe] - self.Firm_Fixed_Cost) / self.Salaire / \
																	   (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe])))


	def firm_hire(self, hire_size_advantage):
		## Firms adjust their
		recruiting = self.Firm_Existe & (self.Firm_Labor_Hiring > self.tol)
		while (self.Free_Labor > self.tol) and np.any(recruiting):
			n_recruiting = np.sum(recruiting)
			ind_rectuitement = np.min(self.Free_Labor / n_recruiting, np.min(self.Firm_New_Labor_Need[recruiting]))
			self.Firm_Labor[self.Firm_Existe, 0] += ind_rectuitement
			self.Free_Labor = np.min(0, self.Free_Labor - ind_rectuitement * n_recruiting)


	def firm_produce(self):
		###Les firmes ayant décidé de rester non durables mettent à jour la quantité qu'elles vont produire
		##Les firmes durables produisent exactement les quantités planifiées
		#select = self.Firm_Existe & np.logical_not(self.Firm_Durable)

		##Apply financial constraint: cannot produce more than what funds allow
		self.Firm_Production[self.Firm_Existe] = np.minimum(self.Firm_Production[self.Firm_Existe],
															np.maximum(0.0,
			  														   self.Firm_Avoir[self.Firm_Existe] - self.Firm_Fixed_Cost) / self.Salaire / \
																	   (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]))

		##Apply resource constraint: cannot produce more than available resources
		tot_n_sus_prod = sum(self.Firm_Production[self.Firm_Existe] * (1.0 - self.Firm_Durable[self.Firm_Existe]))
		if tot_n_sus_prod < self.Resource:
			self.Resource -= tot_n_sus_prod
		else:
			raise ValueError("Not enough resources!")

		self.Prev_Last_Total_Prod = self.Last_Total_Prod
		self.Last_Total_Prod = np.sum(self.Firm_Last_Production[self.Firm_Existe])

		self.Firm_Last_Production[self.Firm_Existe] = self.Firm_Production[self.Firm_Existe]

		self.Firm_Cost[self.Firm_Existe] += self.Firm_Production[self.Firm_Existe] * \
											self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe])

		return self.Firm_Production[self.Firm_Existe]

	def firm_set_price(self):
		self.Firm_Prix[self.Firm_Existe] = np.where(self.Firm_Production[self.Firm_Existe] < self.tol,
													1.0, ##Afin d'éviter une division par zero, dans la ligne suivante on met une condition qui n'a pas d'importance
													(1.0 + self.Firm_Markup) / np.where(self.Firm_Production[self.Firm_Existe] < self.tol, 1.0, self.Firm_Production[self.Firm_Existe]) * \
													self.Firm_Cost[self.Firm_Existe])
													##Dépenses
													#(self.Salaire / self.Firm_Productivite[self.Firm_Existe] * self.Firm_Production[self.Firm_Existe] + \
	      											# self.Taux_Interet * (self.Firm_Credit_Rec_Bank[self.Firm_Existe] + self.Firm_Credit_Rec_Fonds[self.Firm_Existe]) + \
													# self.Firm_Fixed_Cost))

		return self.Firm_Prix[self.Firm_Existe]
		
	def consume(self):
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

		return self.Last_Total_Consumption

	def consume_max_wealth(self, prop_cons=1.0):
		#5.b.Les consommateurs choisissent les produits au hasard et en fonction des prix.
		####Only one round of purchase ?? we can make it with several rounds until income or all firm stocks are depleted
		###WE DO SEVERAL CONSUMPTION ROUNDS ---> TO DO
		self.Last_Total_Consumption = 0.0

		avail_to_consume = prop_cons * self.Avail_Wealth
		##select selling firms
		firm_selling = self.Firm_Existe & (self.Firm_Production > self.tol)
		
		##Selling firms are those whithin 2 stard deviations of the average price
		firm_selling &= self.Firm_Prix - np.mean(self.Firm_Prix[firm_selling]) < 2.0 * np.std(self.Firm_Prix[firm_selling])

		while np.any(firm_selling) & (avail_to_consume > self.tol):
			self.Firm_Demande[firm_selling] = avail_to_consume * self.Firm_Prix[firm_selling]**(-(self.Sens_Prix + 1.0)) / \
											  sum(self.Firm_Prix[firm_selling]**(-self.Sens_Prix))

			###Update income and total consumption
			firm_sales = self.Firm_Prix[firm_selling] * np.minimum(self.Firm_Demande[firm_selling], self.Firm_Production[firm_selling])

			spent_cons = sum(firm_sales)

			avail_to_consume -= spent_cons
			self.Last_Total_Consumption += spent_cons

			###Update firms stocks, assets and selling firms
			self.Firm_Avoir[firm_selling] += firm_sales

			self.Firm_Production[firm_selling] -= np.minimum(self.Firm_Demande[firm_selling], self.Firm_Production[firm_selling])

			firm_selling[firm_selling] &= self.Firm_Production[firm_selling] > self.tol

		self.Avail_Wealth -= prop_cons * self.Avail_Wealth - avail_to_consume

		return self.Last_Total_Consumption

	def firm_pay_bank(self):
		###Ici, il ne s'agit pas de Crédit reçu à proprement parler, mais on stocke les créances vis à vis
		###de la banque dans ce vecteur afin de ne pas en créer de nouveau (performance du code)
		
		###Remboursement de la banque
		###Intérêt des prêts reversés aux consommateurs
		tot_interest = sum(np.minimum(self.Firm_Credit_Rec_Bank[self.Firm_Existe] * self.Taux_Interet,
							np.maximum(self.Firm_Avoir[self.Firm_Existe] - self.Firm_Credit_Rec_Bank[self.Firm_Existe], 0.0)))


		self.Avail_Wealth += tot_interest
		self.Income += tot_interest
		#np.minimum(self.Firm_Credit_Rec_Bank[self.Firm_Existe], np.maximum(self.Firm_Avoir[self.Firm_Existe], 0.0)))
		
		self.Firm_Avoir[self.Firm_Existe] -= (1.0+self.Taux_Interet) * self.Firm_Credit_Rec_Bank[self.Firm_Existe]

		return self.Firm_Avoir[self.Firm_Existe]


	###Remboursement du fonds
	def firm_pay_fund(self):
		tot_fonds_payment = sum(np.minimum((1.0 + self.Taux_Interet) * self.Firm_Credit_Rec_Fonds[self.Firm_Existe],
									   	   np.maximum(self.Firm_Avoir[self.Firm_Existe], 0.0)))

		self.Fonds_Avoir += tot_fonds_payment
		self.Fonds_Yield = 0.0 if self.Fonds_Yield == 0.0 else (tot_fonds_payment - self.Fonds_Yield) / self.Fonds_Yield

		self.Firm_Avoir[self.Firm_Existe] -= (1.0+self.Taux_Interet) * self.Firm_Credit_Rec_Fonds[self.Firm_Existe]

		return self.Firm_Avoir[self.Firm_Existe]


	def cons_withdraw_fund(self, sens_new_info=1.0):
		self.Taux_Retrait_Fonds = sens_new_info * (1.0 - (self.Fonds_Yield + 1.0) / (2.0 + self.Taux_Interet)) + \
								  (1.0 - sens_new_info) * self.Taux_Retrait_Fonds
		withdrawn = self.Taux_Retrait_Fonds * self.Fonds_Avoir
		###Les consommateurs retirent du fonds
		self.Avail_Wealth += withdrawn
		self.Fonds_Avoir -= withdrawn

		return self.Fonds_Avoir


	def firm_pay_salary(self):
		tot_salary = sum(np.minimum(self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] + \
									  self.Firm_Fixed_Cost,
									np.maximum(self.Firm_Avoir[self.Firm_Existe], 
		      								   0.0)))

		##Paiement des salaires
		self.Avail_Wealth += tot_salary
		self.Income += tot_salary
		self.Firm_Avoir[self.Firm_Existe] -= self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] + \
											 self.Firm_Fixed_Cost
		
		return self.Avail_Wealth
		

	def firm_exit(self) -> None:
		###Calculate number of firms exits
		n_exits = np.sum(self.Firm_Existe)
		### Exiting firms
		exiting = self.Firm_Existe[self.Firm_Existe] &\
				  (self.Firm_Avoir[self.Firm_Existe] < self.tol)
		#### Free labor of exiting firms
		self.Free_Labor += np.sum(self.Firm_Labor[exiting, :])
		#### Put labor of exiting firms to 0
		self.Firm_Labor[exiting, :] = 0.0
		###Firmes qui ont un avoir négatif quittent le marché
		self.Firm_Existe[self.Firm_Existe] &= (self.Firm_Avoir[self.Firm_Existe] >= self.tol)
		##To return number of exits
		n_exits -= np.sum(self.Firm_Existe)

		return n_exits
		
	def cons_invest_fund(self):
		###Les consommateurs investissent dans le fonds
		curr_invest_fonds = min(self.Taux_Invest_Fonds * self.Income, self.Avail_Wealth)
		self.Fonds_Avoir += curr_invest_fonds
		self.Avail_Wealth -= curr_invest_fonds

		return self.Fonds_Avoir

	def update_salary(self, m_e):
		if (self.Prev_Last_Total_Prod > 0) & (self.Last_Total_Prod > 0):
			if ((self.Last_Total_Prod / self.Prev_Last_Total_Prod - 1) > self.Labor_Rigidity) |\
			   ((self.Last_Total_Prod / self.Prev_Last_Total_Prod - 1) < -self.Labor_Rigidity):
				self.Salaire *= (self.Last_Total_Prod / self.Prev_Last_Total_Prod)**m_e
				self.Firm_Fixed_Cost *= (self.Last_Total_Prod / self.Prev_Last_Total_Prod)**m_e
		return self.Salaire

	def recover_resource(self):
		self.Resource = min(self.Resource + self.Res_Recovery_Rate * self.Resource * (1.0 - self.Resource / self.Max_Resource),
		      				self.Max_Resource)
		

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

	def firm_total_revenue(self):
		return sum((self.Firm_Last_Production[self.Firm_Existe] - self.Firm_Production[self.Firm_Existe]) * self.Firm_Prix[self.Firm_Existe])

	def non_sustainable_consumed_quantity(self):
		return sum(self.Firm_Last_Production[self.Firm_Existe & np.logical_not(self.Firm_Durable)] - self.Firm_Production[self.Firm_Existe & np.logical_not(self.Firm_Durable)])

	def total_money(self):
		return sum(self.Firm_Avoir[self.Firm_Existe]) + self.Avail_Wealth + self.Fonds_Avoir

	def total_credit(self):
		return sum(self.Firm_Credit_Rec_Bank[self.Firm_Existe] + self.Firm_Credit_Rec_Fonds[self.Firm_Existe])

	def fund_assets(self):
		return self.Fonds_Avoir