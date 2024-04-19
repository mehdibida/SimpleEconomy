# Modèle : est-ce que les banques ont une influence sur la société 
import numpy as np

###Checker si les crédits doivent ou pas aller dans les avoirs des firmes et le faire le cas échéant.

class Economy:
	def __init__(self, 
			  	 Max_Resource, Res_Recovery_Rate, 
			  	 Taux_Entrep, 
				 Sustainability_Interest_Rate_Mult, 
			  	 Prod_Adjust,
				 Init_Wealth,
				 Sustainability_Cost_Mult,
				 N_Firm = 100, tol = 10.0**-6) -> None:
		##resource levels
		self.Resource = Max_Resource
		self.Res_Recovery_Rate = Res_Recovery_Rate
		self.Max_Resource = Max_Resource
				
		self.Sustainability_Cost_Mult = Sustainability_Cost_Mult ## Factor determining the final cost of being sustainable
		self.Sustainability_Interest_Rate_Mult = Sustainability_Interest_Rate_Mult

		#self.Population = Population #Total population
		self.Taux_Entrep = Taux_Entrep #Taux d'entreprenariat: combien on prélève pour la création de nouvelles entreprises
		self.Avail_Wealth = Init_Wealth
		self.Prod_Adjust = Prod_Adjust

		##Current iteration of the economy
		self.Iteration = 0

		##Numerical tolerance of the model (in order to avoid errors due to floating point representation)
		self.tol = tol

		##Caracteristiques des entreprises
		self.Firm_Existe = np.full(N_Firm, False) ##L'entreprise est active.
		self.Firm_Sustainability = np.zeros(N_Firm) ##Sustainability level, between 0 and 1 (1 is 100% sustainable)
		self.Firm_Moment_Creation = np.zeros(N_Firm) ##Iteration at which firm was created
		self.Firm_Production = np.zeros(N_Firm) #Production de chaque entreprise
		self.Firm_Prod_Cost = np.zeros(N_Firm) #Firms current production cost
		self.Firm_Avoir = np.zeros(N_Firm) #Avoir de l'entreprise
		self.Firm_Credit = np.zeros(N_Firm) #Credits demandés
		self.Firm_Profit = np.zeros(N_Firm) ## Firm profit during iteration

		###Random number generator
		self.rng = np.random.default_rng()

	def firm_enter(self, Capital_Dist=np.random.exponential, **dist_kwargs):
		## Capital available to create firms
		n_avail_capital = self.Taux_Entrep * self.Avail_Wealth
		## Initial capital of new firms
		new_firm_capital = np.zeros(np.uint32(np.ceil(n_avail_capital * 1.5)))
		###Sustainability level of newly created firms
		new_firm_sus = np.zeros(np.uint32(np.ceil(n_avail_capital * 1.5)))
		## Capital of entering firms
		n_created = 0
		while (n_avail_capital > self.tol):
			## Extend array if we still need to create firms
			if n_created == len(new_firm_capital):
				new_firm_capital.resize(np.uint32(np.ceil(n_created * 1.5)), refcheck = False)
				new_firm_sus.resize(np.uint32(np.ceil(n_created * 1.5)), refcheck = False)
			## Decide on new firms capital: random initial capital
			new_firm_capital[n_created] = min(Capital_Dist(**dist_kwargs), n_avail_capital)
			## Decide on new firms sustainability levels
			new_firm_sus[n_created] = np.random.uniform(0.0, 1.0)
			### Subtract entrepreneurs and capital from available
			n_avail_capital -= new_firm_capital[n_created]
			n_created += 1

		##Extend vectors if there is no sufficient place for all active firms
		if len(new_firm_capital) > (len(self.Firm_Existe) - sum(self.Firm_Existe)):
			extension_size = n_created
			self.Firm_Existe.resize(len(self.Firm_Existe) + extension_size, refcheck = False)
			self.Firm_Credit.resize(len(self.Firm_Credit) + extension_size, refcheck = False)
			self.Firm_Production.resize(len(self.Firm_Production) + extension_size, refcheck = False)
			self.Firm_Sustainability.resize(len(self.Firm_Sustainability) + extension_size, refcheck = False)
			self.Firm_Prod_Cost.resize(len(self.Firm_Prod_Cost) + extension_size, refcheck = False)
			self.Firm_Profit.resize(len(self.Firm_Profit) + extension_size, refcheck = False)
			self.Firm_Avoir.resize(len(self.Firm_Avoir) + extension_size, refcheck = False)
			self.Firm_Moment_Creation.resize(len(self.Firm_Moment_Creation) + extension_size, refcheck = False)

		###Initialize new firms
		#Liste d'indices dispos qui correspondent aux cases qui vont contenir les nouvelles entreprises
		Indices_Entrants = np.flatnonzero(np.logical_not(self.Firm_Existe))[:n_created] ##Trouver les indices qui sont False
		self.Firm_Existe[Indices_Entrants] = True
		self.Firm_Credit[Indices_Entrants] = 0.0
		self.Firm_Production[Indices_Entrants] = 0.0
		self.Firm_Prod_Cost[Indices_Entrants] = 0.0
		self.Firm_Profit[Indices_Entrants] = 0.0
		self.Firm_Sustainability[Indices_Entrants] = new_firm_sus[:n_created]
		self.Firm_Avoir[Indices_Entrants] = new_firm_capital[:n_created]
		self.Firm_Moment_Creation[Indices_Entrants] = self.Iteration

		return n_created

	def firm_decide_production(self):
		##Firms not making profit diminish their production (to loose less money)
		## Firm making profit raise their production levels
		self.Firm_Production[self.Firm_Existe] *= (1.0 + self.Prod_Adjust)**(np.sign(self.Firm_Profit[self.Firm_Existe]) *
													  						 np.random.uniform(0, 1.0, np.sum(self.Firm_Existe)))

		entrants = self.Firm_Existe & (self.Firm_Moment_Creation == self.Iteration)
		if np.any(entrants):
			self.Firm_Production[entrants] = self.Firm_Avoir[entrants] /\
											 (1.0 + self.Sustainability_Cost_Mult * np.arctanh(self.Firm_Sustainability[entrants])) *\
											 2.0**np.random.uniform(-0.4, 0.4, np.sum(entrants))
		
		return self.Firm_Production

	def firm_pay_dividend(self):
		self.Firm_Prod_Cost[self.Firm_Existe] = self.Firm_Production[self.Firm_Existe] *\
											 	(1.0 + self.Sustainability_Cost_Mult * np.arctanh(self.Firm_Sustainability[self.Firm_Existe]))

		self.Avail_Wealth += np.sum(np.maximum(0.0,
								  			   self.Firm_Avoir[self.Firm_Existe] - 
											   self.Firm_Prod_Cost[self.Firm_Existe]))
		self.Firm_Avoir[self.Firm_Existe] = np.minimum(self.Firm_Prod_Cost[self.Firm_Existe],
													   self.Firm_Avoir[self.Firm_Existe])

	def firm_get_credit(self):
		self.Firm_Prod_Cost[self.Firm_Existe] = self.Firm_Production[self.Firm_Existe] *\
											 	(1.0 + self.Sustainability_Cost_Mult * np.arctanh(self.Firm_Sustainability[self.Firm_Existe]))
		##Firms decide on how much credit they need
		self.Firm_Credit[self.Firm_Existe] = np.maximum(0.0, self.Firm_Prod_Cost[self.Firm_Existe] - self.Firm_Avoir[self.Firm_Existe])
		###Update firms assets
		self.Firm_Avoir[self.Firm_Existe] += self.Firm_Credit[self.Firm_Existe]

		return self.Firm_Credit
		
	def firm_produce_sell(self):
		### Update system resources
		self.Resource -= np.sum(self.Firm_Production[self.Firm_Existe] * (1.0 - self.Firm_Sustainability[self.Firm_Existe]))
		##Determine market price
		market_price = self.Avail_Wealth / np.sum(self.Firm_Production[self.Firm_Existe])
		### Determine firms' profits
		self.Firm_Profit[self.Firm_Existe] = self.Firm_Production[self.Firm_Existe] * market_price -\
											 self.Firm_Prod_Cost[self.Firm_Existe] -\
											 self.Firm_Credit[self.Firm_Existe] * self.Sustainability_Interest_Rate_Mult * (1.0 - self.Firm_Sustainability[self.Firm_Existe])
		### Put wealth to zero
		self.Avail_Wealth = 0
		### Update firms assets
		self.Firm_Avoir[self.Firm_Existe] += market_price * self.Firm_Production[self.Firm_Existe]

		return market_price
	
	def firm_pay_salary(self):
		self.Avail_Wealth += np.sum(self.Firm_Prod_Cost[self.Firm_Existe])
		self.Firm_Avoir[self.Firm_Existe] -= self.Firm_Prod_Cost[self.Firm_Existe]
		
		return self.Avail_Wealth

	def firm_pay_bank(self):
		bank_profit = np.sum(np.minimum(self.Firm_Avoir[self.Firm_Existe],
								        self.Firm_Credit[self.Firm_Existe] * 
							 			(1 + self.Sustainability_Interest_Rate_Mult * (1.0 - self.Firm_Sustainability[self.Firm_Existe])))) -\
					  np.sum(self.Firm_Credit[self.Firm_Existe])
							 
		self.Avail_Wealth += max(bank_profit, 0.0)
		
		self.Firm_Avoir[self.Firm_Existe] -= self.Firm_Credit[self.Firm_Existe] *\
							 				 (1 + self.Sustainability_Interest_Rate_Mult * (1.0 - self.Firm_Sustainability[self.Firm_Existe]))

		return bank_profit

	def firm_exit(self) -> None:
		###To Calculate number of firms exits
		n_exits = np.sum(self.Firm_Existe)
		### Firms with assets below 0 exit the market
		self.Firm_Existe[self.Firm_Existe] &= (self.Firm_Avoir[self.Firm_Existe] >= self.tol)
		##To return number of exits
		n_exits -= np.sum(self.Firm_Existe)

		return n_exits

	def recover_resource(self):
		self.Resource = min(self.Resource + self.Res_Recovery_Rate * self.Resource * (1.0 - self.Resource / self.Max_Resource),
		      				self.Max_Resource)
		return self.Resource
		
	def increment_iteration(self) -> None:
		self.Iteration += 1

	####Economy information methods
	def firms_count(self):
		return sum(self.Firm_Existe)

	def gross_product(self):
		return self.Firm_Production[self.Firm_Existe]
	
	def average_production_sustainability(self):
		return np.average(self.Firm_Sustainability[self.Firm_Existe],
						  weights = self.Firm_Production[self.Firm_Existe])

	def total_money(self):
		return sum(self.Firm_Avoir[self.Firm_Existe]) + self.Avail_Wealth
