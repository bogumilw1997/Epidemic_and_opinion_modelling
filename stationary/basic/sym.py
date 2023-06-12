import sys
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from json import load
import networkx as nx
import copy
from collections import Counter
import seaborn as sns
from scipy.optimize import curve_fit
import math
from tqdm import tqdm
from simulation import simulation

plt.rcParams["figure.figsize"] = [14, 8]
plt.rcParams['font.size'] = '13'

with open("praca_mgr\programy\wersja2_infections\params.json") as f:
    parameters = load(f)
    
N = parameters['N'] # liczba wezlow
m = parameters['m'] # m dla sieci BA
T = parameters['T'] # liczba kroków symulacji

realizations = parameters['realizations'] # liczba realizacji symulacji
v_step = parameters['v_step'] # liczba kroków symulacji opinii względem epidemii

p0_inf = parameters['p0_inf']
p0_op_minus = parameters['p0_op_minus']
p0_op_plus = parameters['p0_op_plus']

#beta = parameters['beta'] # prawd. zarażenia
omega = parameters['omega'] # skut. szczepionki
gamma = parameters['gamma'] # prawd. kwarantanny
mu = parameters['mu'] # prawd. wyzdrowienia (1 - kappa)
kappa = 1 - mu # prawd. śmierci

beta_min = parameters['beta_min'] # prawd. zarażenia min
beta_max = parameters['beta_max'] # prawd. zarażenia max
beta_interval = parameters['beta_interval']

graph_path = parameters['graph_path'] # sciezka zapisu wykresow
save_path = parameters['save_path'] # sciezka zapisu danych

mode = parameters['mode']

g = nx.barabasi_albert_graph(N, m)

df = pd.DataFrame()
df_ = pd.DataFrame()

print('Symulacja w trybie: ' + mode)
print('N: ' + str(N))
print('omega: ' + str(omega))
print('gamma: ' + str(gamma))
print('mu: ' + str(mu))
print('v_step: ' + str(v_step))

# sym = simulation(g,g, 0.1, omega, gamma, mu, v_step, kappa, mode)
# sym.init_states(p0_inf, p0_op_minus, p0_op_plus)

# print(sym.inf_period_list)
# sys.exit()

for beta in tqdm(np.arange(beta_min,beta_max + 0.001,beta_interval)):

    for realization in tqdm(range(realizations)):

        sym = simulation(g,g, beta, omega, gamma, mu, v_step, kappa, mode)
        sym.init_states(p0_inf, p0_op_minus, p0_op_plus)

        infections_list = np.zeros(T)
        opinion_list = np.zeros(T)
        vacc_list = np.zeros(T)
        total_vacc_list = np.zeros(T)
        
        infections_list[0] = sym.get_inf_number()
        opinion_list[0] = sym.get_mean_opinion()
        vacc_list[0] = sym.get_vacc_number()
        total_vacc_list[0] = sym.get_total_vacc_number()
        
        for t in tqdm(range(1, T)):
            
            sym.do_one_sim_step()
            
            infections_list[t] = sym.get_inf_number()
            opinion_list[t] = sym.get_mean_opinion()
            vacc_list[t] = sym.get_vacc_number()
            total_vacc_list[t] = sym.get_total_vacc_number()
            
        df_['inf'] = infections_list
        df_['opinion'] = opinion_list
        df_['vacc'] = vacc_list
        df_['total_vacc'] = total_vacc_list
        df_['beta'] = beta
        df_['t'] = np.arange(T)
        
        df = pd.concat([df, df_], ignore_index = True)

# if os.path.exists(save_path + 'infections_opinions.csv'):
#   os.remove(save_path + 'infections_opinions.csv')
  
df.to_csv(save_path + 'infections_opinions.csv')
