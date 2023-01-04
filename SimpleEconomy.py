# Modèle : est-ce que les banques ont une influence sur la société ?

import numpy as np
import matplotlib.pyplot as plt



#0 System_Sustainable_Firms #nombre d'entreprise durables, initialement zero
#1 System_PIB #PIB initial, l'économie commence à 0.
#2 System_Money sum(np.maximum(Firm_Avoir[Firm_Existe], 0)) + Income #somme monétaire , l'économie commence à 0.
#3 System_Credit sum(Firm_Credit_Rec[Firm_Existe]) #nombre de crédit, personne n'a de crédit initialement.
class Economy:

    def __init__(self, Taux_Interet, Prime_D, Firm_Markup, Prod_Ajust, Fonds_Dividende, Taux_Invest_Fonds, Init_Income, Sens_Prix, Init_Prop_Durable=0.0, N_Firm = 100, tol = 10.0**-6) -> None:
        self.Prop_Durable = Init_Prop_Durable  #Proportion minimum de prêt durable, initialement à zero
        self.Taux_Interet = Taux_Interet #Le taux d'intérêt des Fonds_Avoirs/crédits
        self.Epargne = 0.0
        #self.Qualité_Durable = 2.0 #Qualité des produits vendus par les firmes durables, qualité des non-durables à 1
        #Salaires
        self.Prime_D = 1.0 + Prime_D #Prime pour les salaires durables (facteur) #Il est en hashtag dans le modèle V1 : pourquoi ?
        self.Salaire = 1.0 #Salaire non durable
        self.Income = Init_Income
        self.Firm_Markup = Firm_Markup
        self.Fonds_Dividende = Fonds_Dividende
        self.Taux_Invest_Fonds = Taux_Invest_Fonds
        self.Sens_Prix = Sens_Prix
        self.Prod_Ajust = Prod_Ajust

        ##Numerical tolerance of the model (in order to avoid errors due to floating point representation)
        self.tol = tol


        ##Caracteristiques des entreprises
        self.Firm_Existe = np.full(N_Firm, False) ##L'entreprise est active.
        self.Firm_Durable = np.full(N_Firm, False) ##Durable ou pas.
        self.Firm_New_Durable = np.full(N_Firm, False) ##Firmes devenant durables.

        self.Firm_Moment_Creation = np.zeros(N_Firm) ##Iteration at which firm was created

        self.Firm_Production = np.zeros(N_Firm) #Production de chaque entreprise
        self.Firm_Profit = np.zeros(N_Firm) #Niveau des profits faits par l'entreprise
        self.Firm_Avoir = np.zeros(N_Firm) #Avoir de l'entreprise
        self.Firm_Credit_Dem = np.zeros(N_Firm) #Credits demandés
        self.Firm_Credit_Rec_Bank = np.zeros(N_Firm) #Credits reçu par l'entreprise (octroyé par la banque)
        self.Firm_Credit_Rec_Fonds = np.zeros(N_Firm)
        self.Firm_Invest_Fonds = np.zeros(N_Firm) #Création d'un Fonds sur une ponction des profits après du versement du salaire
        self.Firm_Demande = np.zeros(N_Firm) #Quantités demandées à chacune de entreprises
        self.Firm_Prix = np.ones(N_Firm) #Prix des entreprises

        ##Caractéristiques des fonds
        self.Fonds_Avoir = 0.0  #Taille du fonds (initialization à zero)


    def set_prop_durable(self, prop) -> None:
        self.Prop_Durable = prop

    def firm_enter(self, N_Entrants, iteration) -> None:
        ##Extend vectors if there is no sufficient place for all active firms
        if N_Entrants > (len(self.Firm_Existe) - sum(self.Firm_Existe)):
            extension_size = N_Entrants - sum(np.logical_not(self.Firm_Existe))
            self.Firm_Existe.resize(len(self.Firm_Existe) + extension_size, refcheck = False)
            self.Firm_Durable.resize(len(self.Firm_Durable) + extension_size, refcheck = False)
            self.Firm_New_Durable.resize(len(self.Firm_New_Durable) + extension_size, refcheck = False)
            self.Firm_Production.resize(len(self.Firm_Production) + extension_size, refcheck = False)
            self.Firm_Profit.resize(len(self.Firm_Profit) + extension_size, refcheck = False)
            self.Firm_Avoir.resize(len(self.Firm_Avoir) + extension_size, refcheck = False)
            self.Firm_Credit_Dem.resize(len(self.Firm_Credit_Dem) + extension_size, refcheck = False)
            self.Firm_Credit_Rec_Bank.resize(len(self.Firm_Credit_Rec_Bank) + extension_size, refcheck = False)
            self.Firm_Credit_Rec_Fonds.resize(len(self.Firm_Credit_Rec_Fonds) + extension_size, refcheck = False)
            self.Firm_Invest_Fonds.resize(len(self.Firm_Invest_Fonds) + extension_size, refcheck = False)
            self.Firm_Demande.resize(len(self.Firm_Demande) + extension_size, refcheck = False)
            self.Firm_Prix.resize(len(self.Firm_Prix) + extension_size, refcheck = False)
            self.Firm_Moment_Creation.resize(len(self.Firm_Moment_Creation) + extension_size, refcheck = False)
            self.Firm_Prix[np.logical_not(self.Firm_Existe)] = 1.0
        ###Initialize new firms
        #Liste d'indices dispos qui correspondent aux cases qui vont contenir les nouvelles entreprises
        Indices_New = np.flatnonzero(np.logical_not(self.Firm_Existe))[:N_Entrants] ##Trouver les indices qui son False
        self.Firm_Production[Indices_New] = 0.0
        self.Firm_Profit[Indices_New] = 0.0
        self.Firm_Avoir[Indices_New] = 0.0
        self.Firm_Credit_Dem[Indices_New] = 0.0
        self.Firm_Credit_Rec_Bank[Indices_New] = 0.0
        self.Firm_Credit_Rec_Fonds[Indices_New] = 0.0
        self.Firm_Invest_Fonds[Indices_New] = 0.0
        self.Firm_Existe[Indices_New] = True
        self.Firm_Moment_Creation[Indices_New] = iteration
        
    def firm_decide_production(self, iteration):
        new_entrants = np.flatnonzero(self.Firm_Existe & (self.Firm_Moment_Creation == iteration))
        incumbents = np.flatnonzero(self.Firm_Existe & (self.Firm_Moment_Creation < iteration))
        
        if len(incumbents) > 0:
             ##2.b : Les entreprises existantes ajustent la production par rapport aux ventes précédentes. Prod_Sensibilite entre 0 et 1 et correspond à la sensibilité des entrepeneurs.
            self.Firm_Production[incumbents] = self.Firm_Production[incumbents] + self.Prod_Ajust * (self.Firm_Demande[incumbents] - self.Firm_Production[incumbents])
        
        if len(new_entrants) > 0:
            if len(incumbents) > 0:
                self.Firm_Production[new_entrants] = np.random.uniform(0.75, 1.25, size = len(new_entrants)) #* np.mean(self.Firm_Production[self.Firm_Existe])
            else:
                self.Firm_Production[new_entrants] = np.random.uniform(size = len(new_entrants))

    def firm_credit_dem(self):

        self.Firm_Credit_Dem[self.Firm_Existe] = np.maximum(self.Salaire * (1 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe] - self.Firm_Avoir[self.Firm_Existe], 0)
       
        ### Initialisation
        #Fonds  : "invetsissement" correspond à la qté totale de fonds disponible dans le fonds.
        #self.Fonds_Avoir += np.sum(self.Firm_Invest_Fonds[Firm_Existe])

        if self.Fonds_Avoir >= sum(self.Firm_Credit_Dem[self.Firm_Existe]):
            self.Firm_Credit_Rec_Fonds[self.Firm_Existe] = self.Firm_Credit_Dem[self.Firm_Existe]
            self.Firm_Credit_Dem[self.Firm_Existe] = 0.0
        else: 
            self.Firm_Credit_Rec_Fonds[self.Firm_Existe] = self.Fonds_Avoir * self.Firm_Credit_Dem[self.Firm_Existe] / sum(self.Firm_Credit_Dem[self.Firm_Existe])
            #Chaque entreprise reçoivent (Firm_Credit_Rec_Fonds) proportionnellement à ce qu'elles ont demandés (FIrm_Credit_Dem)
            #Investit = Firm_Credit_Rec_Fonds[Firm_Existe] < Firm_Credit_Dem[Firm_Existe] 
            self.Firm_Credit_Dem[self.Firm_Existe] -= self.Firm_Credit_Rec_Fonds[self.Firm_Existe]         
            self.Fonds_Avoir = 0.0

        ### CREDITS BANCAIRES ### ##Le firmes durables reçoivent les crédits voulus
            self.Firm_Credit_Rec_Bank[self.Firm_Existe & self.Firm_Durable] = self.Firm_Credit_Dem[self.Firm_Existe & self.Firm_Durable]

        #3.d. Selon la différence entre la proportion effective et le taux exogène imposé, les banques font une offre de crédits aux entreprises non durable selon un processus en « fontaine ». 
            Demande_Credit_N_durable = sum(self.Firm_Credit_Dem[self.Firm_Existe & np.logical_not(self.Firm_Durable)])

        #Crédit total non durable à donner
            if self.Prop_Durable > 0:
                Offre_Credit_N_Durable = (1 - self.Prop_Durable) / self.Prop_Durable * sum(self.Firm_Credit_Rec_Bank[self.Firm_Existe & self.Firm_Durable])
            else:
                Offre_Credit_N_Durable = Demande_Credit_N_durable + self.tol

            if Offre_Credit_N_Durable >= Demande_Credit_N_durable:
                self.Firm_Credit_Rec_Bank[self.Firm_Existe & np.logical_not(self.Firm_Durable)] = self.Firm_Credit_Dem[self.Firm_Existe & np.logical_not(self.Firm_Durable)]
            else:
                self.Firm_Credit_Rec_Bank[self.Firm_Existe & np.logical_not(self.Firm_Durable)] = Offre_Credit_N_Durable * \
                                                        self.Firm_Credit_Dem[self.Firm_Existe & np.logical_not(self.Firm_Durable)] / Demande_Credit_N_durable
            #Recoit est le tableau des indices de ceux qui ont recu moins que ce qu'ils ont demandé (et peuvent donc encore recevoir)
            #Recoit = Firm_Credit_Rec_Bank[Firm_Existe & np.logical_not(Firm_Durable)] < Firm_Credit_Dem[Firm_Existe & np.logical_not(Firm_Durable)]

        #3.f.   Les entreprises évaluent ensuite ces deux offres ensemble et les compare au profit qu’ils feraient s’ils devenaient durables.
        ### EQUATION SUR PAPIER MEHDI 13 AVRIL.        
        ##partie avant 1+Taux_Interet: Nouveau crédit nécessaire pour produire la projection vu la prime durable
            self.Firm_New_Durable[self.Firm_Existe] = (- (self.Salaire * ((self.Prime_D - self.Firm_Markup) * (self.Firm_Avoir[self.Firm_Existe] + self.Firm_Credit_Dem[self.Firm_Existe]) + \
                                                       self.Taux_Interet * (1.0 + self.Prime_D) * (self.Firm_Credit_Dem[self.Firm_Existe] - self.Firm_Credit_Rec_Bank[self.Firm_Existe])) + \
                                                       self.Firm_Markup * (self.Firm_Avoir[self.Firm_Existe] + self.Firm_Credit_Rec_Bank[self.Firm_Existe]) * (1.0 + self.Prime_D)) > 0.0) & \
                                                       np.logical_not(self.Firm_Durable[self.Firm_Existe])
            

                                                ### Je comprends pas bien pourquoi il y a des multiplications dans la première ligne 139. 



            ###Les firmes nouvellement devenues durables réévaluent leur besoin en financement (car les salaires vont changer)
            ###L'addition de (Prime_D * Firm_Avoir) est là pour compenser le fait que Firm_Credit_Dem prend déjà en compte les avoirs
            self.Firm_Credit_Dem[self.Firm_New_Durable] = self.Firm_Credit_Dem[self.Firm_New_Durable] * (1+self.Prime_D) + self.Prime_D * self.Firm_Avoir[self.Firm_New_Durable]
            #Les firmes ayant reçu moins d'argent que demandé deviennent durables, elles reçoivent donc le financement demandé
            self.Firm_Credit_Rec_Bank[self.Firm_New_Durable] = self.Firm_Credit_Dem[self.Firm_New_Durable]
        
            ## On change leur indice de durabilité en firmes durables
            self.Firm_Durable[self.Firm_Existe] = self.Firm_Durable[self.Firm_Existe] | self.Firm_New_Durable[self.Firm_Existe]
            
            ###Les firmes ayant décidé de rester non durables mettent à jour la quantité qu'elles vont produire
            ##Les firmes durables produisent exactement les quantités planifiées
            select = self.Firm_Existe & np.logical_not(self.Firm_Durable)
            self.Firm_Production[select] = np.minimum(self.Firm_Production[select], 
                                                      (self.Firm_Credit_Rec_Bank[select] + self.Firm_Credit_Rec_Fonds[select] + self.Firm_Avoir[select]) / self.Salaire)

                                                 ### Pourquoi divisé par salaire, et pas "-" salaire?
            ###On ajoute temporairement les prets aux Avoirs des entreprises
            self.Firm_Avoir[self.Firm_Existe] += self.Firm_Credit_Rec_Bank[self.Firm_Existe] + self.Firm_Credit_Rec_Fonds[self.Firm_Existe] 
            ###Adjust production with respect to assets
            self.Firm_Production[self.Firm_Existe] = np.minimum(self.Firm_Production[self.Firm_Existe],
                                                                (self.Firm_Avoir[self.Firm_Existe] - self.Taux_Interet * (self.Firm_Credit_Rec_Bank[self.Firm_Existe] + self.Firm_Credit_Rec_Fonds[self.Firm_Existe])) / \
                                                                self.Salaire / (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]))

    def firm_set_price(self) -> None:
        self.Firm_Prix[self.Firm_Existe] = np.where(self.Firm_Production[self.Firm_Existe] < self.tol,
                                                    1.0,
                                                    (1 + self.Firm_Markup) / self.Firm_Production[self.Firm_Existe] * \
                                                    (self.Salaire * self.Firm_Production[self.Firm_Existe] + self.Taux_Interet * (self.Firm_Credit_Rec_Bank[self.Firm_Existe] + self.Firm_Credit_Rec_Fonds[self.Firm_Existe])))
        
    def consume(self) -> None:
        #5.b.Les consommateurs choisissent les produits au hasard et en fonction des prix.
        ####Only one round of purchase ?? we can make it with several rounds until income or all firm stocks are depleted
        ###WE DO SEVERAL CONSUMPTION ROUNDS ---> TO DO
        firm_selling = self.Firm_Existe & (self.Firm_Production > 0.0)
        self.Firm_Demande[firm_selling] = self.Income * self.Firm_Prix[firm_selling]**(-(self.Sens_Prix + 1.0)) / \
                                sum(self.Firm_Prix[firm_selling]**(-self.Sens_Prix))

        ####On augmente les avoirs de la firme par ses revenus de la vente (et non des dividendes)
        self.Firm_Avoir[firm_selling] += self.Firm_Prix[firm_selling] * np.minimum(self.Firm_Demande[firm_selling], self.Firm_Production[firm_selling])
        self.Income -= sum(self.Firm_Prix[firm_selling] * np.minimum(self.Firm_Demande[firm_selling], self.Firm_Production[firm_selling]))

    def make_loan_payment(self) -> None:
        ###Ici, il ne s'agit pas de Crédit reçu à proprement parler, mais on stocke les créances vis à vis
        ###de la banque dans ce vecteur afin de ne pas en créer de nouveau (performance du code)
        self.Firm_Credit_Rec_Bank[self.Firm_Existe] *= (1.0+self.Taux_Interet)

        #print("Avoir avant paiement ", self.Firm_Avoir[self.Firm_Existe])
        ###On soustrait des créances ce que les entreprises peuvent rembourser grâce à leurs profits
        ###venant des ventes (et non des dividendes du fonds)
        for i in range(len(self.Firm_Existe)):
            if self.Firm_Existe[i]:
                if self.Firm_Credit_Rec_Bank[i] > self.Firm_Avoir[i]:
                    self.Firm_Credit_Rec_Bank[i] -= max(self.Firm_Avoir[i], 0.0)
                    self.Firm_Avoir[i] = 0.0
                else:
                    self.Firm_Avoir[i] -= max(self.Firm_Credit_Rec_Bank[i], 0.0)
                    self.Firm_Credit_Rec_Bank[i] = 0.0

        #print("Avoir après paiement des banques", self.Firm_Avoir[self.Firm_Existe])

        ###Calcul du bilan lié au fonds
        ###Firm avoir peut etre négatif et redevenir positif grace aux dividendes
        ###Paiment du fonds de la part des firmes ayant encore de l'argent
        Fonds_Retour = sum(np.minimum(self.Firm_Credit_Rec_Fonds[self.Firm_Existe]*(1.0+self.Taux_Interet), np.maximum(self.Firm_Avoir[self.Firm_Existe], 0.0)))
                 ### Si le maximum est de Firm_Avoir, les entreprises ne peuvent pas aller en négatif et donc ne peuvent pas faire faillite à cause du fond : juste? 
                 ### Il devrait y avoir une condition que si elles ne peuvent pas payer l'entier de la somme, elles font faillites. =>> existe mais plus bas. À voir si ça joue?
        Fonds_Profit = Fonds_Retour - sum(self.Firm_Credit_Rec_Fonds[self.Firm_Existe])
        
        ###Créances vis à vis du fonds: Ici, il ne s'agit pas de Crédit reçu à proprement parler, mais on stocke les créances vis à vis
        ###du fonds dans ce vecteur afin de ne pas en créer de nouveau (performance du code)
        self.Firm_Credit_Rec_Fonds[self.Firm_Existe] *= 1.0 + self.Taux_Interet

        ####Paiement au fonds et mise à jour des avoirs des firmes
        for i in range(len(self.Firm_Existe)):
            if self.Firm_Existe[i]:
                if self.Firm_Credit_Rec_Fonds[i] > self.Firm_Avoir[i]:
                    self.Firm_Credit_Rec_Fonds[i] -= max(self.Firm_Avoir[i], 0.0)
                    self.Firm_Avoir[i] = 0.0
                else:
                    self.Firm_Avoir[i] -= max(self.Firm_Credit_Rec_Fonds[i], 0.0)
                    self.Firm_Credit_Rec_Fonds[i] = 0.0

        #print("Avoir après paiement du fonds", self.Firm_Avoir[self.Firm_Existe])

        ##On augmente les avoirs du fonds après paiement des créances
        self.Fonds_Avoir += Fonds_Retour - np.maximum(Fonds_Profit, 0.0) * self.Fonds_Dividende
        ###Firmes reçoivent le dividende du fonds
        tot_invest_fonds = sum(self.Firm_Invest_Fonds[self.Firm_Existe])
        if (tot_invest_fonds > 0.0) & (self.Fonds_Dividende > 0.0):
            self.Firm_Avoir[self.Firm_Existe] += self.Firm_Invest_Fonds[self.Firm_Existe] / tot_invest_fonds * self.Fonds_Dividende * Fonds_Profit

        ### Pas sûr de bien comprendre la division ici. 

        ###Firmes ayant encore des dettes auprès de la banque payent leur créances
        self.Firm_Avoir[self.Firm_Existe] -= self.Firm_Credit_Rec_Bank[self.Firm_Existe]
        self.Firm_Credit_Rec_Bank[self.Firm_Existe] = 0.0
        #print("Avoir après paiement des banques", self.Firm_Avoir[self.Firm_Existe])
        ###Firmes ayant encore des dettes auprès du fonds et ayant encore un avoir positif lui payent leur créances
        self.Fonds_Avoir += sum(np.minimum(self.Firm_Credit_Rec_Fonds[self.Firm_Existe], np.maximum(self.Firm_Avoir[self.Firm_Existe], 0.0)))
        ###Soustraction des dettes payées au fonds à l'avoir de la firme (pour voir s'il y a faillite)
        self.Firm_Avoir[self.Firm_Existe] -= self.Firm_Credit_Rec_Fonds[self.Firm_Existe]
        self.Firm_Credit_Rec_Fonds[self.Firm_Existe] = 0.0
        ###Firmes qui ont un avoir négatif quittent le marché
        self.Firm_Existe[self.Firm_Existe] = self.Firm_Avoir[self.Firm_Existe] >= 0.0


    def pay_salary(self) -> None:
        self.Income += sum(self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe])
        self.Firm_Avoir[self.Firm_Existe] -= self.Salaire * (1.0 + self.Prime_D * self.Firm_Durable[self.Firm_Existe]) * self.Firm_Production[self.Firm_Existe]
        ###Firmes qui ont un avoir négatif quittent le marché
        #self.Firm_Existe[self.Firm_Existe] = self.Firm_Avoir[self.Firm_Existe] >= 0.0

    def firm_invest_fund(self) -> None:
        ###Les firmes investissent dans le fonds
        self.Firm_Invest_Fonds[self.Firm_Existe] += self.Firm_Avoir[self.Firm_Existe] * self.Taux_Invest_Fonds
        self.Fonds_Avoir += sum(self.Firm_Avoir[self.Firm_Existe] * self.Taux_Invest_Fonds)
        ###Pourquoi pas de Firm_Existe à Taux_Invest_Fonds?
        self.Firm_Avoir[self.Firm_Existe] *= (1.0 - self.Taux_Invest_Fonds)


    ####Economy information methods
    def nb_firms(self):
        return sum(self.Firm_Existe)

    def nb_sustainable_firms(self):
        return sum(self.Firm_Durable[self.Firm_Existe])

    def gross_product(self):
        return self.Income

    def total_money(self):
        return sum(self.Firm_Avoir[self.Firm_Existe]) + self.Income + self.Fonds_Avoir

    def total_credit(self):
        return sum(self.Firm_Credit_Rec_Bank[self.Firm_Existe] + self.Firm_Credit_Rec_Fonds[self.Firm_Existe])

    def fund_assets(self):
        return self.Fonds_Avoir


T_Max = 250


eco = Economy(Taux_Interet = 0.1, Prime_D = 0.1, Firm_Markup = 0.0, Prod_Ajust = 0.3, 
            Fonds_Dividende = 0.1, Taux_Invest_Fonds = 0.00, Init_Income = 5.0, Sens_Prix = 1.5, 
            N_Firm = T_Max)

gdp = np.zeros(T_Max)

for t in range(T_Max):
    if t/T_Max > 10:
        eco.set_prop_durable(t/T_Max)
    if t > 2:
        n_entrants = np.random.randint(5, 10) * int(1 + round(gdp[t-1]))
    else:
        n_entrants = np.random.randint(5, 10)
    eco.firm_enter(n_entrants, t)
    eco.firm_decide_production(t)
    #print("---------Step ", t, " ----------------")
    eco.firm_credit_dem()
    #print("Firm Rec Fonds: ", eco.Firm_Credit_Rec_Fonds[eco.Firm_Existe])
    #print(eco.Firm_Avoir)
    eco.firm_set_price()
    eco.pay_salary()
    eco.consume()
    #print(eco.Firm_Avoir) 
    eco.make_loan_payment()
    #print(eco.Firm_Avoir)
    #print(eco.Firm_Avoir) 
    eco.firm_invest_fund()
    #print(eco.Firm_Avoir)
    gdp[t] = eco.gross_product()


##ajouter des épargnants qui veulent maximiser leur rémunération par leur épargne, épargne investie

plt.plot(gdp)

sum(eco.Firm_Avoir[eco.Firm_Existe])

plt.scatter(eco.Firm_Moment_Creation[eco.Firm_Existe], eco.Firm_Avoir[eco.Firm_Existe])




def main():
    T_Max = 1000

    eco = Economy(Taux_Interet = 0.05, Prime_D = 0.1, Firm_Markup = 0.1, Prod_Ajust = 0.5, 
                Fonds_Dividende = 0.1, Taux_Invest_Fonds = 0.05, Init_Income = 1.0, Sens_Prix = 1.0, 
                N_Firm = T_Max)

    gdp = np.zeros(T_max)

    for t in range(T_Max):
        if t/T_Max > 0.2:
            eco.set_prop_durable(t/T_Max)
        n_entrants = np.random.randint(0, 5)
        eco.firm_enter(n_entrants)
        eco.firm_decide_production(n_entrants)
        eco.firm_credit_dem()
        eco.firm_set_price()
        eco.consume()
        eco.make_loan_payment()
        eco.pay_salary()
        eco.firm_invest_fund()     
        gdp[t] = eco.gross_product()


gdp





    # System_Total_Firm en premier, ensuite Sustainable, ensuite PIB, ensuite Money, ensuite Credit, ensuite, Fonds.
        #0 System_Total_Firms
        #1 System_Sustainable_Firms #nombre d'entreprise durables, initialement zero
        #2 System_PIB #PIB initial, l'économie commence à 0.
        #3 System_Money sum(np.maximum(Firm_Avoir[Firm_Existe], 0)) + Income #somme monétaire , l'économie commence à 0.
        #4 System_Credit sum(Firm_Credit_Rec[Firm_Existe]) #nombre de crédit, personne n'a de crédit initialement.
        #5 System_Fonds
 
plt.plot([t for t in range(T_Max)], [res[1][i] / res[0][i] for i in range(T_Max)])
plt.plot([t for t in range(T_Max)], [res[0][i] for i in range(T_Max)])


plt.show()

#plt.plot(res[1][:T_Max])

#plt.plot(res[1][:T_Max] / res[0][:T_Max])

#plt.plot(res[0][1:T_Max] - res[0][:T_Max-1])

#plt.plot(res[2][:T_Max])

#plt.plot(res[3][:T_Max])

#plt.plot(res[0][1:T_Max] - res[0][:T_Max-1])

#plt.plot(np.log(res[1][1:T_Max] / res[1][:(T_Max-1)]))

#plt.plot(np.log(res[1][100:] / res[1][:300]))

#plt.plot(res[5][:T_Max]) #system Fonds









def model(Temps_Max, Taux_Interet, Prop_Durable_Max, Prime_D, Firm_Markup, Prod_Ajust, Fonds_Dividende, Taux_Invest_Fonds, Init_Income, Sens_Prix):
    #####Caracteristiques du systeme
    Prop_Durable = 0  #Proportion minimum de prêt durable, initialement à zero
    Fonds_Avoir = 0  #Proportion INITIALE des fonds avoir
    Taux_Interet = 0.05 #Le taux d'intérêt des Fonds_Avoirs/crédits
    Epargne = 0
    Qualité_Durable = 2.0 #Qualité des produits vendus par les firmes durables, qualité des non-durables à 1
    #Salaires
    #Prime_D = 1.2 #Prime pour les salaires durables (facteur) #Il est en hashtag dans le modèle V1 : pourquoi ?
    Salaire = 1 #Salaire non durable

    System_Total_Firms = np.zeros(Temps_Max) #nombre total d'entreprises
    System_Sustainable_Firms = np.zeros(Temps_Max) #nombre d'entreprise durables, initialement zero
    System_PIB = np.zeros(Temps_Max) #PIB initial, l'économie commence à 0.
    #System_Volume_Firms_Sus = np.zeros(Temps_Max) #volume des entreprises durable, commence à 0
    #System_Volume_Firms_UnSus = np.zeros(Temps_Max) #volume des entreprises non durable, commence à 0
    System_Money = np.zeros(Temps_Max) #somme monétaire , l'économie commence à 0.
    System_Credit = np.zeros(Temps_Max) #nombre de crédit, personne n'a de crédit initialement.
    System_Fonds = np.zeros(Temps_Max) #taille du fonds d'Fonds_Avoir ponctionné sur les entreprises. POURQUOI FONDS AVOIR : REDONDANT ?
    #System_Fonds_Avoir = np.zeros(Temps_Max) ##nombre d'Fonds_Avoir, personne n'a d'Fonds_Avoirs.
    #System_Prix_Firms_Sus = np.zeros(Temps_Max)
    #System_Prix_Firms_UnSus = np.zeros(Temps_Max)
    
    ###Allocation de Vecteurs initiaux
    N_Firm = Temps_Max

    ##Caracteristiques des entreprises
    Firm_Existe = np.full(N_Firm, False) ##L'entreprise est active.
    Firm_Durable = np.full(N_Firm, False) ##Durable ou pas.
    Firm_New_Durable = np.full(N_Firm, False) ##Firmes devenant durables.

    Firm_Production = np.zeros(N_Firm) #Production de chaque entreprise
    Firm_Profit = np.zeros(N_Firm) #Niveau des profits faits par l'entreprise
    Firm_Avoir = np.zeros(N_Firm) #Avoir de l'entreprise
    Firm_Credit_Dem = np.zeros(N_Firm) #Credits demandés
    Firm_Credit_Rec_Bank = np.zeros(N_Firm) #Credits reçu par l'entreprise (octroyé par la banque)
    Firm_Credit_Rec_Fonds = np.zeros(N_Firm)
    Firm_Invest_Fonds = np.zeros(N_Firm) #Création d'un Fonds sur une ponction des salariés au moment du versement du salaire
    Firm_Demande = np.zeros(N_Firm) #Quantités demandées à chacune de entreprises
    Firm_Prix = np.zeros(N_Firm) #Prix des entreprises

    ##Caracteristiques des ménages
    Income = Init_Income #HS = Household
    
    ##Iterations
    for t in range(Temps_Max):

        if (t >= (Temps_Max * 0.2)):
            Prop_Durable = min((t/Temps_Max - 0.2), Prop_Durable_Max) 
        ##Changements
        #1. Les nouvelles entreprises se créent.
        #1.a :  nombre d'entreprises entrantes au hasard
        N_Entrants = np.random.randint(0, 5)

        ##Extend vectors if there is no sufficient place for all active firms
        if N_Entrants > sum(np.logical_not(Firm_Existe)):
            Firm_Existe.resize(len(Firm_Existe) + N_Firm, refcheck = False)
            Firm_Durable.resize(len(Firm_Durable) + N_Firm, refcheck = False)
            Firm_New_Durable.resize(len(Firm_New_Durable) + N_Firm, refcheck = False)
            Firm_Production.resize(len(Firm_Production) + N_Firm, refcheck = False)
            Firm_Profit.resize(len(Firm_Profit) + N_Firm, refcheck = False)
            Firm_Avoir.resize(len(Firm_Avoir) + N_Firm, refcheck = False)
            Firm_Credit_Dem.resize(len(Firm_Credit_Dem) + N_Firm, refcheck = False)
            Firm_Credit_Rec_Bank.resize(len(Firm_Credit_Rec_Bank) + N_Firm, refcheck = False)
            Firm_Credit_Rec_Fonds.resize(len(Firm_Credit_Rec_Fonds) + N_Firm, refcheck = False)
            Firm_Invest_Fonds.resize(len(Firm_Invest_Fonds) + N_Firm, refcheck = False)
            Firm_Demande.resize(len(Firm_Demande) + N_Firm, refcheck = False)
            Firm_Prix.resize(len(Firm_Prix) + N_Firm, refcheck = False)
        
        #Liste d'indices dispos qui correspondent aux cases qui vont contenir les nouvelles entreprises
        Indices_New = np.flatnonzero(np.logical_not(Firm_Existe))[:N_Entrants] #le logical_not ici est pour trouver les "false" de Firm_Existe.

        #2. Evaluation de la Production
        ##2.a : les entreprises entrantes évaluent leur production en fonction de la moyenne de production des entreprises existantes
        if np.any(Firm_Existe):
            Firm_Production[Indices_New] = np.mean(Firm_Demande[Firm_Existe]) * np.random.uniform(size = len(Indices_New))
        else:
            Firm_Production[Indices_New] = np.random.uniform(size = len(Indices_New))
        
        ##2.b : Les entreprises existantes ajustent la production par rapport aux ventes précédentes. Prod_Sensibilite entre 0 et 1 et correspond à la sensibilité des entrepeneurs.
        Firm_Production[Firm_Existe] = Firm_Production[Firm_Existe] + Prod_Ajust * (Firm_Demande[Firm_Existe] - Firm_Production[Firm_Existe])
        
        #Mettre en **existe** les entreprises nouvellement créées
        Firm_Existe[Indices_New] = True

        #3. Demandes de crédit

        #3.a.   Les entreprises evaluent les crédits nécessaires
        Firm_Credit_Dem[Firm_Existe] = np.maximum(Salaire * (1 + Prime_D * Firm_Durable[Firm_Existe]) * Firm_Production[Firm_Existe] - Firm_Avoir[Firm_Existe], 0)
       
        ### Initialisation
        #Fonds  : "invetsissement" correspond à la qté totale de fonds disponible dans le fonds.
       
        Fonds_Avoir = Fonds_Avoir + np.sum(Firm_Invest_Fonds[Firm_Existe])
        if Fonds_Avoir >= sum(Firm_Credit_Dem[Firm_Existe]):
            Firm_Credit_Rec_Fonds[Firm_Existe] = Firm_Credit_Dem[Firm_Existe]
            Firm_Credit_Dem[Firm_Existe] = 0.0
        else: 
            Firm_Credit_Rec_Fonds[Firm_Existe] = Fonds_Avoir * Firm_Credit_Dem[Firm_Existe] / sum(Firm_Credit_Dem[Firm_Existe])
            #Chaque entreprise reçoivent (Firm_Credit_Rec_Fonds) proportionnellement à ce qu'elles ont demandés (FIrm_Credit_Dem)
            #Investit = Firm_Credit_Rec_Fonds[Firm_Existe] < Firm_Credit_Dem[Firm_Existe] 
            Firm_Credit_Dem[Firm_Existe] -= Firm_Credit_Rec_Fonds[Firm_Existe]         
            Fonds_Avoir -= sum(Firm_Credit_Rec_Fonds[Firm_Existe])

        ### CREDITS BANCAIRES ### ##Le firmes durables reçoivent les crédits voulus
            Firm_Credit_Rec_Bank[Firm_Existe & Firm_Durable] = Firm_Credit_Dem[Firm_Existe & Firm_Durable]

        #3.d. Selon la différence entre la proportion effective et le taux exogène imposé, les banques font une offre de crédits aux entreprises non durable selon un processus en « fontaine ». 
            Demande_Credit_N_durable = sum(Firm_Credit_Dem[Firm_Existe & np.logical_not(Firm_Durable)])

        #Crédit total non durable à donner
            if Prop_Durable > 0:
                Offre_Credit_N_Durable = (1 - Prop_Durable) / Prop_Durable * sum(Firm_Credit_Rec_Bank[Firm_Existe & Firm_Durable])
            else:
                Offre_Credit_N_Durable = 2 * Demande_Credit_N_durable

            if Offre_Credit_N_Durable >= Demande_Credit_N_durable:
                Firm_Credit_Rec_Bank[Firm_Existe & np.logical_not(Firm_Durable)] = Firm_Credit_Dem[Firm_Existe & np.logical_not(Firm_Durable)]
            else:
                Firm_Credit_Rec_Bank[Firm_Existe & np.logical_not(Firm_Durable)] = Offre_Credit_N_Durable * \
                                                                                   Firm_Credit_Dem[Firm_Existe & np.logical_not(Firm_Durable)] / Demande_Credit_N_durable
            #Recoit est le tableau des indices de ceux qui ont recu moins que ce qu'ils ont demandé (et peuvent donc encore recevoir)
            #Recoit = Firm_Credit_Rec_Bank[Firm_Existe & np.logical_not(Firm_Durable)] < Firm_Credit_Dem[Firm_Existe & np.logical_not(Firm_Durable)]

        #3.f.   Les entreprises évaluent ensuite ces deux offres ensemble et les compare au profit qu’ils feraient s’ils devenaient durables.
        ### EQUATION SUR PAPIER MEHDI 13 AVRIL.        
        ##partie avant 1+Taux_Interet: Nouveau crédit nécessaire pour produire la projection vu la prime durable
            Firm_New_Durable[Firm_Existe] = (- (Salaire * ((Prime_D - Firm_Markup) * (Firm_Avoir[Firm_Existe] + Firm_Credit_Dem[Firm_Existe]) + \
                                                       Taux_Interet * (1.0 + Prime_D) * (Firm_Credit_Dem[Firm_Existe] - Firm_Credit_Rec_Bank[Firm_Existe])) + \
                                             Firm_Markup * (Firm_Avoir[Firm_Existe] + Firm_Credit_Rec_Bank[Firm_Existe]) * (1.0 + Prime_D)) > 0.0) & \
                                            np.logical_not(Firm_Durable[Firm_Existe])
            

                                                ### Je comprends pas bien pourquoi il y a des multiplications dans la première ligne 139. 



            ###Les firmes nouvellement devenues durables réévaluent leur besoin en financement (car les salaires vont changer)
            ###L'addition de (Prime_D * Firm_Avoir) est là pour compenser le fait que Firm_Credit_Dem prend déjà en compte les avoirs
            Firm_Credit_Dem[Firm_New_Durable] = Firm_Credit_Dem[Firm_New_Durable] * (1+Prime_D) + Prime_D * Firm_Avoir[Firm_New_Durable]
            #Les firmes ayant reçu moins d'argent que demandé deviennent durables, elles reçoivent donc le financement demandé
            Firm_Credit_Rec_Bank[Firm_New_Durable] = Firm_Credit_Dem[Firm_New_Durable]
        
            ## On change leur indice de durabilité en firmes durables
            Firm_Durable[Firm_Existe] = Firm_Durable[Firm_Existe] | Firm_New_Durable[Firm_Existe]
            
            ###Les firmes ayant décidé de rester non durables mettent à jour la quantité qu'elles vont produire
            ##Les firmes durables produisent exactement les quantités planifiées
            select = Firm_Existe & np.logical_not(Firm_Durable)
            Firm_Production[select] = np.minimum(Firm_Production[select], 
                                                 (Firm_Credit_Rec_Bank[select] + Firm_Credit_Rec_Fonds[select] + Firm_Avoir[select]) / Salaire)

                                                 ### Pourquoi divisé par salaire, et pas "-" salaire?

        ###On ajoute temporairement les prets aux Avoirs des entreprises
            Firm_Avoir[Firm_Existe] += Firm_Credit_Rec_Bank[Firm_Existe] + Firm_Credit_Rec_Fonds[Firm_Existe] 

        #4. Production (Rien n'a changé, chaque firme va produire exactement ce qu'elle voulait produire)
        #5.a.   Les entreprises mettent en vente sur le marché les produits.
        ###La prime de salaire durable n'est pas incluse dans le prix car les firmes durables vont vendre au prix du non durable
        Firm_Prod = Firm_Existe & (Firm_Production > 0.0)
        Firm_Prix[Firm_Prod] = (1 + Firm_Markup) / Firm_Production[Firm_Prod] * \
                               (Salaire * Firm_Production[Firm_Prod] + Taux_Interet * (Firm_Credit_Rec_Bank[Firm_Prod] + Firm_Credit_Rec_Fonds[Firm_Prod]))


        #5.b.Les consommateurs choisissent les produits au hasard et en fonction des prix.
        Firm_Demande[Firm_Existe] = Income * Firm_Prix[Firm_Existe]**(-(Sens_Prix + 1)) / \
                                    sum(Firm_Prix[Firm_Existe]**(-Sens_Prix))

        ####On augmente les avoirs de la firme par ses revenus de la vente (et non des dividendes)
        Firm_Avoir[Firm_Existe] += Firm_Prix[Firm_Existe] * np.minimum(Firm_Demande[Firm_Existe], Firm_Production[Firm_Existe])

        ###Ici, il ne s'agit pas de Crédit reçu à proprement parler, mais on stocke les créances vis à vis
        ###de la banque dans ce vecteur afin de ne pas en créer de nouveau (performance du code)
        Firm_Credit_Rec_Bank[Firm_Existe] *= (1+Taux_Interet)
        
        ###On soustrait des créances ce que les entreprises peuvent rembourser grâce à leurs profits
        ###venant des ventes (et non des dividendes du fonds)
        Firm_Credit_Rec_Bank[Firm_Existe] -= np.minimum(Firm_Credit_Rec_Bank[Firm_Existe], Firm_Avoir[Firm_Existe])

        ####Paiment des prêts et intérêts aux banques
        Firm_Avoir[Firm_Existe] -= Firm_Credit_Rec_Bank[Firm_Existe]

        ###Calcul du bilan lié au fonds
        ###Firm avoir peut etre négatif et redevenir positif grace aux dividendes
        ###Paiment du fonds de la part des firmes ayant encore de l'argent
        Fonds_Retour = sum(np.minimum(Firm_Credit_Rec_Fonds[Firm_Existe]*(1+Taux_Interet), np.maximum(Firm_Avoir[Firm_Existe], 0))) 
                 ### Si le maximum est de Firm_Avoir, les entreprises ne peuvent pas aller en négatif et donc ne peuvent pas faire faillite à cause du fond : juste? 
                 ### Il devrait y avoir une condition que si elles ne peuvent pas payer l'entier de la somme, elles font faillites. =>> existe mais plus bas. À voir si ça joue?
        Fonds_Profit = Fonds_Retour - sum(Firm_Credit_Rec_Fonds[Firm_Existe])
        
        ###Créances vis à vis du fonds: Ici, il ne s'agit pas de Crédit reçu à proprement parler, mais on stocke les créances vis à vis
        ###du fonds dans ce vecteur afin de ne pas en créer de nouveau (performance du code)
        Firm_Credit_Rec_Fonds[Firm_Existe] *= 1+Taux_Interet
        ### Mettre à jour les avoirs de la firme après le paiement du fonds
        Firm_Avoir[Firm_Existe] -= Firm_Credit_Rec_Fonds[Firm_Existe]
        ###On met à jour ce que les entreprises doivent au fonds
        Firm_Credit_Rec_Fonds[Firm_Existe] = - np.minimum(0.0, Firm_Avoir[Firm_Existe])  

        ##On augmente les avoirs du fonds après paiement des créances
        Fonds_Avoir += Fonds_Retour - np.maximum(Fonds_Profit, 0.0) * Fonds_Dividende
        ###Firmes reçoivent le dividende du fonds
        tot_invest_fonds = sum(Firm_Invest_Fonds[Firm_Existe])
        if (tot_invest_fonds > 0) & (Fonds_Dividende > 0):
            Firm_Avoir[Firm_Existe] += Firm_Invest_Fonds[Firm_Existe] / tot_invest_fonds * Fonds_Dividende * Fonds_Profit

                        ### Pas sûr de bien comprendre la division ici. 

        ###Firmes ayant encore des dettes auprès de la banque payent leur créances
        Firm_Avoir[Firm_Existe] -= Firm_Credit_Rec_Bank[Firm_Existe]
        ###Firmes ayant encore des dettes auprès du fonds et ayant encore un avoir positif lui payent leur créances
        Fonds_Avoir += sum(np.minimum(Firm_Credit_Rec_Fonds[Firm_Existe], np.maximum(Firm_Avoir[Firm_Existe], 0)))
        ###Soustraction des dettes payées au fonds à l'avoir de la firme (pour voir s'il y a faillite)
        Firm_Avoir[Firm_Existe] -= Firm_Credit_Rec_Fonds[Firm_Existe]

        ###Firmes qui ont un avoir négatif quittent le marché
        Firm_Existe[Firm_Existe] = Firm_Avoir[Firm_Existe] >= 0

        ###Les firmes investissent dans le fonds
        Firm_Invest_Fonds[Firm_Existe] += Firm_Avoir[Firm_Existe] * Taux_Invest_Fonds
        Fonds_Avoir += sum(Firm_Avoir[Firm_Existe] * Taux_Invest_Fonds) 
                ###Pourquoi pas de Firm_Existe à Taux_Invest_Fonds?
        Firm_Avoir[Firm_Existe] *= (1-Taux_Invest_Fonds)
        
        #4.c :  les entreprises paient les salaires aux consommateurs.
        ## Revenus des conommateurs: Salaires payés + Revenus des taux t'intérêt + dividendes
        ##Mise a jour du revenu pour l'itération suivante
        Income = Salaire * sum((1 + Prime_D * Firm_Durable[Firm_Existe]) * Firm_Production[Firm_Existe])

        ###6.d.	Si une entreprise se retrouvent en situation d’incapacité de paiement de toutes ces créances, elle fait faillite après avoir « vider ses comptes ». Une entreprise qui fait faillite disparaît du marché et ne peut plus revenir.
        #Firm_Existe[Firm_Existe] = (Firm_Avoir[Firm_Existe] >= 0)

        
        ###Mise a jour des variables d'etat du systeme
        System_Total_Firms[t] = sum(Firm_Existe)
        System_Sustainable_Firms[t] = sum(Firm_Durable[Firm_Existe]) #nombre d'entreprise durables, initialement zero
        System_PIB[t] = Income #PIB initial, l'économie commence à 0.
        System_Money[t] = sum(Firm_Avoir[Firm_Existe]) + Income + Fonds_Avoir #somme monétaire , l'économie commence à 0.
        System_Credit[t] = sum(Firm_Credit_Rec_Bank[Firm_Existe] + Firm_Credit_Rec_Fonds[Firm_Existe]) #nombre de crédit, personne n'a de crédit initialement. // ###ADD Firm_Credit_Rec_Fonds[Firm_Existe]
        System_Fonds[t] = Fonds_Avoir     

        #INITIALEMENT ICI.
        ###6.d.	Si une entreprise se retrouvent en situation d’incapacité de paiement de toutes ces créances, elle fait faillite après avoir « vider ses comptes ». Une entreprise qui fait faillite disparaît du marché et ne peut plus revenir.
        #Firm_Existe[Firm_Existe] = (Firm_Avoir[Firm_Existe] >= 0)

        #System_Fonds_Avoir[t] = np.zeros(Temps_Max)
    return (System_Total_Firms, System_Sustainable_Firms, System_PIB, System_Money, System_Credit, System_Fonds)



