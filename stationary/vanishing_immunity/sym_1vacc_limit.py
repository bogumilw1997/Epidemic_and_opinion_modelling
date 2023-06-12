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

plt.rcParams["figure.figsize"] = [14, 8]
plt.rcParams['font.size'] = '15'
plt.rcParams['lines.linewidth'] = '1.5'
plt.rcParams['lines.markersize'] = '9'
plt.rcParams["figure.autolayout"] = True

with open("praca_mgr/programy5_2fala/mod_zach_odp/params.json") as f:
    parameters = load(f)
    
N = parameters['N'] # liczba wezlow
m = parameters['m'] # m dla sieci BA
T = parameters['T'] # liczba kroków symulacji

realizations = parameters['realizations'] # liczba realizacji symulacji
v_step = parameters['v_step'] # liczba kroków symulacji opinii względem epidemii

p0_inf = parameters['p0_inf']
p0_op_minus = parameters['p0_op_minus']
p0_op_plus = parameters['p0_op_plus']

beta = parameters['beta'] # prawd. zarażenia
gamma = parameters['gamma'] # prawd. kwarantanny
omega = parameters['omega'] # skut. szczepionki
mu = parameters['mu'] # prawd. wyzdrowienia (1 - kappa)
kappa = 1 - mu # prawd. śmierci

vacc_death_prob = parameters['vacc_death_prob']

beta_min = parameters['beta_min'] # prawd. zarażenia min
beta_max = parameters['beta_max'] # prawd. zarażenia max
beta_interval = parameters['beta_interval']

graph_path = parameters['graph_path'] # sciezka zapisu wykresow
save_path_local = parameters['save_path_local'] # sciezka zapisu danych
save_path_serwer = parameters['save_path_serwer']

waning_time = parameters['waning_time'] # czas zaniku odporności

mode = parameters['mode']

vacc_start = parameters['vacc_start']
vacc_limit = parameters['vacc_limit']

# fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.plot(vacc_curve1)
# ax1.set_xlabel('t')
# ax1.set_ylabel('vaccinated agents')
# ax1.set_title('dawka1')

# ax2.plot(vacc_curve2)
# ax2.set_xlabel('t')
# ax2.set_ylabel('vaccinated agents')
# ax2.set_title('dawka2')

# plt.show()

# sys.exit()

df = pd.DataFrame()
df_ = pd.DataFrame()

vacc1 = 0.03
vacc2 = 0.06
vacc3 = 0.09
vacc_limit = vacc1

beta1 = 0.15
beta2 = 0.2

from simulation import simulation

for beta in tqdm([beta1, beta2]):
    
    for realization in tqdm(range(realizations)):
        
        g1 = nx.barabasi_albert_graph(N, m)
        g2 = nx.connected_watts_strogatz_graph(N, 6, 0.15)
        
        sym = simulation(g1,g2, beta, omega, gamma, mu, v_step, mode, waning_time, vacc_death_prob, vacc_start, vacc_limit)
        sym.init_states(p0_inf, p0_op_minus, p0_op_plus)

        infections_list = np.zeros(T)
        opinion_list = np.zeros(T)
        vacc_list = np.zeros(T)
        susc_list = np.zeros(T)
        quar_list = np.zeros(T)
        recov_list = np.zeros(T)
        dead_list = np.zeros(T)
        total_vacc_list = np.zeros(T)
        total_inf_list = np.zeros(T)
        
        infections_list[0] = sym.get_inf_number()
        opinion_list[0] = sym.get_mean_opinion()
        vacc_list[0] = sym.get_vacc_number()
        susc_list[0] = sym.get_susc_number()
        quar_list[0] = sym.get_quar_number()
        recov_list[0] = sym.get_recov_number()
        dead_list[0] = sym.get_dead_number()
        total_vacc_list[0] = sym.get_total_vacc_number()
        total_inf_list[0] = sym.get_total_inf_number()
        
        for t in tqdm(range(1, T)):
            
            sym.do_one_sim_step(t)
            
            infections_list[t] = sym.get_inf_number()
            opinion_list[t] = sym.get_mean_opinion()
            vacc_list[t] = sym.get_vacc_number()
            susc_list[t] = sym.get_susc_number()
            quar_list[t] = sym.get_quar_number()
            recov_list[t] = sym.get_recov_number()
            dead_list[t] = sym.get_dead_number()
            total_vacc_list[t] = sym.get_total_vacc_number()
            total_inf_list[t] = sym.get_total_inf_number()
            
        df_['inf'] = infections_list
        df_['opinion'] = opinion_list
        df_['vacc'] = vacc_list
        df_['susc'] = susc_list
        df_['quar'] = quar_list
        df_['recov'] = recov_list
        df_['dead'] = dead_list
        
        df_['total_vacc'] = total_vacc_list
        df_['total_inf'] = total_inf_list
        df_['vacc limit'] = vacc_limit
        df_['beta'] = beta
        df_['network'] = 'WS'
        df_['t'] = np.arange(T)
        
        df = pd.concat([df, df_], ignore_index = True)

del g2

from simulation_full_graph import simulation

for beta in tqdm([beta1, beta2]):
    
    for realization in tqdm(range(realizations)):
        
        g1 = nx.barabasi_albert_graph(N, m)
        
        sym = simulation(g1, beta, omega, gamma, mu, v_step, mode, waning_time, vacc_death_prob, vacc_start, vacc_limit)
        sym.init_states(p0_inf, p0_op_minus, p0_op_plus)

        infections_list = np.zeros(T)
        opinion_list = np.zeros(T)
        vacc_list = np.zeros(T)
        susc_list = np.zeros(T)
        quar_list = np.zeros(T)
        recov_list = np.zeros(T)
        dead_list = np.zeros(T)
        total_vacc_list = np.zeros(T)
        total_inf_list = np.zeros(T)
        
        infections_list[0] = sym.get_inf_number()
        opinion_list[0] = sym.get_mean_opinion()
        vacc_list[0] = sym.get_vacc_number()
        susc_list[0] = sym.get_susc_number()
        quar_list[0] = sym.get_quar_number()
        recov_list[0] = sym.get_recov_number()
        dead_list[0] = sym.get_dead_number()
        total_vacc_list[0] = sym.get_total_vacc_number()
        total_inf_list[0] = sym.get_total_inf_number()
        
        for t in tqdm(range(1, T)):
            
            sym.do_one_sim_step(t)
            
            infections_list[t] = sym.get_inf_number()
            opinion_list[t] = sym.get_mean_opinion()
            vacc_list[t] = sym.get_vacc_number()
            susc_list[t] = sym.get_susc_number()
            quar_list[t] = sym.get_quar_number()
            recov_list[t] = sym.get_recov_number()
            dead_list[t] = sym.get_dead_number()
            total_vacc_list[t] = sym.get_total_vacc_number()
            total_inf_list[t] = sym.get_total_inf_number()
            
        df_['inf'] = infections_list
        df_['opinion'] = opinion_list
        df_['vacc'] = vacc_list
        df_['susc'] = susc_list
        df_['quar'] = quar_list
        df_['recov'] = recov_list
        df_['dead'] = dead_list
        
        df_['total_vacc'] = total_vacc_list
        df_['total_inf'] = total_inf_list
        df_['vacc limit'] = vacc_limit
        df_['beta'] = beta
        df_['network'] = 'full graph'
        df_['t'] = np.arange(T)
        
        df = pd.concat([df, df_], ignore_index = True)
        
df.to_csv(save_path_local + 't_300_bety.csv')