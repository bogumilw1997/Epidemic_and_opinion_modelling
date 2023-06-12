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

with open("/home/bwierzchowski/COVID_modeling/programy/new_model/both_ADN/params1.json") as f:
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
save_path_serwer = parameters['save_path_serwer'] # sciezka zapisu danych

df = pd.DataFrame()
df_ = pd.DataFrame()

from Simulation_ADN_independant import Simulation

phi_list = np.arange(0.0,0.21, step=0.01)
eta_list = np.flip(np.arange(10,41, step=2))

max_heatmap_matrix = np.zeros((len(eta_list), len(phi_list)))
time_heatmap_matrix = np.zeros((len(eta_list), len(phi_list)))

for eta_n, eta in enumerate(eta_list):

    for phi_n, phi in enumerate(phi_list):
        
        inf_max = np.zeros(realizations)
        inf_max_time = np.zeros(realizations)
        
        for realization in (range(realizations)):
            
            sym = Simulation(N, m, beta, omega, gamma, mu, phi, kappa, v_step, eta, epsilon, w)
            sym.init_states(p0_inf, p0_op_minus, p0_op_plus)

            infections_list = np.zeros(T)
            
            infections_list[0] = sym.get_inf_number()

            for t in range(1, T):
                
                sym.do_one_sim_step()

                infections_list[t] = sym.get_inf_number()

            inf_max[realization] = np.max(infections_list)
            inf_max_time[realization] = np.argmax(infections_list)
            
        max_heatmap_matrix[eta_n, phi_n] = np.mean(inf_max)
        time_heatmap_matrix[eta_n, phi_n] = np.mean(inf_max_time)
        
df = pd.DataFrame(max_heatmap_matrix)

df.to_csv(save_path_serwer + 'piki_max_heatmap_beta03.csv')

df = pd.DataFrame(time_heatmap_matrix)

df.to_csv(save_path_serwer + 'piki_time_heatmap_beta03.csv')

params_dict = {"eta":eta_list, "phi":phi_list}

df = pd.DataFrame({k:pd.Series(v) for k,v in params_dict.items()})

df.to_csv(save_path_serwer + 'piki_heatmap_beta03_params.csv')