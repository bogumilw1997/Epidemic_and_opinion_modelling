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

with open("praca_mgr\programy7_simple/both_ADN\params.json") as f:
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
gamma = parameters['gamma'] # prawd. wyzdrowienia
omega = parameters['omega'] # skut. szczepionki
mu = parameters['mu'] # prawd. powrotu
phi = parameters['phi'] 
kappa = parameters['kappa'] # prawd. resetu opinii

eta = parameters["eta"]
epsilon = parameters["epsilon"]
w = parameters["w"]

save_path_local = parameters['save_path_local'] # sciezka zapisu danych

df = pd.DataFrame()
df_ = pd.DataFrame()

from Simulation_ADN_lower_eta_inf import Simulation

d_list = [1, 3, 5]

for d in tqdm(d_list):
    
    for realization in tqdm(range(realizations)):
        
        sym = Simulation(N, m, beta, omega, gamma, mu, phi, kappa, v_step, eta, epsilon, w, d)
        sym.init_states(p0_inf, p0_op_minus, p0_op_plus)

        infections_list = np.zeros(T)
        opinion_list = np.zeros(T)
        vacc_list = np.zeros(T)
        total_vacc_list = np.zeros(T)
        
        plus_list = np.zeros(T)
        minus_list = np.zeros(T)
        zero_list = np.zeros(T)
        
        total_vacc_list[0] = sym.get_total_vacc_number()
        infections_list[0] = sym.get_inf_number()
        opinion_list[0] = sym.get_mean_opinion()
        vacc_list[0] = sym.get_vacc_number()
        
        plus_list[0] = sym.get_opinion_plus()
        minus_list[0] = sym.get_opinion_minus()
        zero_list[0] = sym.get_opinion_zero()
        
        for t in tqdm(range(1, T)):
            
            sym.do_one_sim_step()

            infections_list[t] = sym.get_inf_number()
            opinion_list[t] = sym.get_mean_opinion()
            vacc_list[t] = sym.get_vacc_number()
            total_vacc_list[t] = sym.get_total_vacc_number()

            plus_list[t] = sym.get_opinion_plus()
            minus_list[t] = sym.get_opinion_minus()
            zero_list[t] = sym.get_opinion_zero()
        
        df_['inf'] = infections_list
        df_['opinion'] = opinion_list
        df_['vacc'] = vacc_list
        df_['total_vacc'] = total_vacc_list
        
        df_['plus'] = plus_list
        df_['minus'] = minus_list
        df_['zero'] = zero_list
        
        df_['d'] = d
        df_['t'] = np.arange(T)
        
        df = pd.concat([df, df_], ignore_index = True)

df.to_csv(save_path_local + 'ADN_independant_lower_inf_eta.csv')