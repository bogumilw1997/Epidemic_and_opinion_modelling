from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import copy
from collections import Counter
import seaborn as sns
from scipy.optimize import curve_fit
import math
import random

class simulation():
    
    opinion_states = np.array([-1, 0, 1])
    epidemic_states = np.array(['S', 'I', 'R', 'Q', 'D', 'V'])
    
    def __init__(self, g1, g2, beta, omega, gamma, mu, v_step, mode, waning_time, vacc_death_prob, vacc_start, vacc_limit):
        
        self.epidemic_network = copy.deepcopy(g1)
        
        self.N = g1.number_of_nodes()
        self.v_step = v_step
        
        self.waning_time = waning_time
        
        self.nodes_list = np.array(g1.nodes())
        
        self.vacc_death_prob = vacc_death_prob
        self.infected_while_vacc = np.zeros(self.N)
        
        # varray = np.array([vacc_limit * _ for _ in range(50)])
        # varray[varray > 1.0] = 1.0
        # vacc_curve = np.concatenate((np.zeros(vacc_start), varray))
        
        self.vacc_start = vacc_start
        self.daily_vacc_limit = int(vacc_limit * self.N)

        if mode == 'random':
            
            self.allocation = np.random.permutation(self.nodes_list)
            self.nodes_allocation_dict = dict(zip(self.nodes_list, self.allocation))
            
            self.opinion_network = nx.relabel_nodes(g2, self.nodes_allocation_dict)
            
        elif mode == 'equal':
            
            self.opinion_network = copy.deepcopy(g2)
        
        self.epidemic_degrees = np.array(list(dict(self.epidemic_network.degree).values()))        
        self.opinion_degrees = np.array([self.opinion_network.degree[_] for _ in range(self.N)])
        
        self.beta = beta
        self.omega = omega
        self.gamma = gamma
        self.mu = mu
        self.kappa = 1 - mu
        
        self.inf_time_list = np.zeros(self.N)
        self.ever_vaccinated = np.zeros(self.N)
        self.time_immune = np.zeros(self.N)
        self.ever_infected = np.zeros(self.N)
        
    def get_pearson_corr(self):
        
        mean_epidemic_deg = np.mean(self.epidemic_degrees)
        mean_opinion_deg = np.mean(self.opinion_degrees)
        
        std_epidemic_deg = np.std(self.epidemic_degrees)
        std_opinion_deg = np.std(self.opinion_degrees)
        
        mean_product = np.mean(self.epidemic_degrees*self.opinion_degrees)
        
        r = (mean_product - mean_epidemic_deg*mean_opinion_deg)/(std_epidemic_deg*std_opinion_deg)
        
        return r
        
    def init_states(self, p0_inf, p0_op_minus, p0_op_plus):
        
        self.inf_period_list = np.array([math.ceil(_) for _ in np.abs(np.random.normal(10, 5, self.N))])
        self.stubbornness = np.array([min(_, 1) for _ in np.abs(np.random.normal(0, 0.333, self.N))])
        self.immunation_time = np.array([math.ceil(_) for _ in np.abs(np.random.normal(self.waning_time, 5, self.N))])
        
        self.epidemic_state = np.random.choice(self.epidemic_states[:2], self.N, p=[1- p0_inf, p0_inf])
        self.opinion_state =  np.random.choice(self.opinion_states, self.N, p=[p0_op_minus, 1 - p0_op_plus - p0_op_minus,p0_op_plus])
        
        self.new_epidemic_state = copy.deepcopy(self.epidemic_state)
        self.new_opinion_state = copy.deepcopy(self.opinion_state)
        
        self.ever_infected[self.epidemic_state == 'I'] = 1
        
        if self.vacc_start == 0:
        
            potential_vacc_nodes = np.logical_and(np.logical_and(self.epidemic_state == 'S', self.opinion_state == 1), self.ever_vaccinated == 0)
            potential_vacc_nodes_number = int(np.sum(potential_vacc_nodes))
            
            if potential_vacc_nodes_number > 0:
                
                if potential_vacc_nodes_number > self.daily_vacc_limit:
                
                    temp_state = np.full(potential_vacc_nodes_number, 'S')
            
                    vacc_nodes = np.random.choice(potential_vacc_nodes_number, self.daily_vacc_limit, replace=False)
                    temp_state[vacc_nodes] = 'V'
                    
                    self.epidemic_state[potential_vacc_nodes] = temp_state
                    self.ever_vaccinated[self.epidemic_state == 'V'] = 1
                    
                else:
                    self.epidemic_state[potential_vacc_nodes] = np.full(potential_vacc_nodes_number, 'V')
                    self.ever_vaccinated[self.epidemic_state == 'V'] = 1
        
        
    def get_inf_number(self):
        return Counter(self.epidemic_state)['I']/self.N
    
    def get_vacc_number(self):
        return Counter(self.epidemic_state)['V']/self.N
    
    def get_susc_number(self):
        return Counter(self.epidemic_state)['S']/self.N

    def get_quar_number(self):
        return Counter(self.epidemic_state)['Q']/self.N
    
    def get_recov_number(self):
        return Counter(self.epidemic_state)['R']/self.N
    
    def get_dead_number(self):
        return Counter(self.epidemic_state)['D']/self.N
    
    def get_total_inf_number(self):
        return np.mean(self.ever_infected)
    
    def get_total_vacc_number(self):
        return np.mean(self.ever_vaccinated)
    
    def get_mean_opinion(self):
        return np.mean(self.opinion_state[self.epidemic_state != 'D'])
    
    def stubborness_dist_graph(self):
        
        g = sns.histplot(self.stubbornness,stat='probability', kde=True)
        g.set(xlabel=r'$\lambda_i$', ylabel=r"$P(\lambda_i)$")
        
        plt.show()
        plt.close()
    
    def degree_distribution_graph(self, first_points, graph_path):
        
        def power_law(x, a, b):
            return a*np.power(x, b)
        
        k1_sorted = sorted(Counter(np.array(list(dict(self.epidemic_network.degree).values()))).items())
        k1 = [i[0] for i in k1_sorted]
        p_k1 = [i[1]/self.N for i in k1_sorted]
        
        k2_sorted = sorted(Counter(np.array(list(dict(self.opinion_network.degree).values()))).items())
        k2 = [i[0] for i in k2_sorted]
        p_k2 = [i[1]/self.N for i in k2_sorted]
        
        if first_points == 'all':
            pars1, cov1 = curve_fit(f=power_law, xdata=k1, ydata=p_k1, p0=[0, 0], bounds=(-np.inf, np.inf))
            pars2, cov2 = curve_fit(f=power_law, xdata=k2, ydata=p_k2, p0=[0, 0], bounds=(-np.inf, np.inf))
            x1 = k1
            y1 = power_law(x1, pars1[0], pars1[1])
            x2 = k2
            y2 = power_law(x2, pars2[0], pars2[1])
        else:
            pars1, cov1 = curve_fit(f=power_law, xdata=k1[:first_points], ydata=p_k1[:first_points], p0=[0, 0], bounds=(-np.inf, np.inf))
            pars2, cov2 = curve_fit(f=power_law, xdata=k2[:first_points], ydata=p_k2[:first_points], p0=[0, 0], bounds=(-np.inf, np.inf))
            x1 = k1[:first_points]
            y1 = power_law(x1, pars1[0], pars1[1])
            x2 = k2[:first_points]
            y2 = power_law(x2, pars2[0], pars2[1])
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Rozkład stopni węzłów')
        g1 = sns.scatterplot(x = k1, y = p_k1, ax = ax1, color='b')
        sns.lineplot(x = x1, y = y1, ax = ax1, color='r')
        g1.set(xscale="log", yscale="log", xlabel=r'$k$', ylabel=r"$P(k)$", title = 'Sieć epidemii')
        ax1.legend(labels = ["dane",r"fit $\gamma$ = " + f'{np.round(pars1[1],2)}'])
        
        g2 = sns.scatterplot(x = k2, y = p_k2, ax = ax2, color='b')
        sns.lineplot(x = x2, y = y2, ax = ax2, color='r')
        g2.set(xscale="log", yscale="log", xlabel=r'$k$', ylabel=r"$P(k)$", title = 'Sieć opinii')
        ax2.legend(labels = ["dane",r"fit $\gamma$ = " + f'{np.round(pars2[1],2)}'])
        
        plt.savefig(graph_path + 'rozklad_wezlow.png')
        plt.show()
        plt.close()
        
        #return [pars1[1], pars2[1]]
    
    def check_epidemic(self, node):
        
        if self.epidemic_state[node] == 'S':
            
            neighbours = np.array(self.epidemic_network[node])
            
            if neighbours.shape[0] > 0:
                    
                    i_neighbours = neighbours[self.epidemic_state[neighbours] == 'I']
                    
                    i_count = i_neighbours.shape[0]
                    
                    if i_count > 0:
                        
                        infection_prob = np.random.random_sample(size = i_count)
                        
                        if all(i_p >= self.beta for i_p in infection_prob):
                            pass
                        else:
                            self.ever_infected[node] = 1
                            self.time_immune[node] = 0
                            self.inf_time_list[node] = 0
                            quarantine_prob = np.random.random()
                            if quarantine_prob >= self.gamma:
                                self.new_epidemic_state[node] = 'I'
                                
                            else:
                                self.new_epidemic_state[node] = 'Q'
                
        elif self.epidemic_state[node] == 'V':
            
            neighbours = np.array(self.epidemic_network[node])
            
            if neighbours.shape[0] > 0:
                    
                    i_neighbours = neighbours[self.epidemic_state[neighbours] == 'I']
                    
                    i_count = i_neighbours.shape[0]
                    
                    if i_count > 0:   
                        
                        infection_prob = np.random.random_sample(size = i_count)
                        
                        if all(i_p >= (self.beta * (1 - self.omega)) for i_p in infection_prob):
                            pass
                        else:
                            self.ever_infected[node] = 1
                            self.time_immune[node] = 0
                            self.inf_time_list[node] = 0
                            self.infected_while_vacc[node] = 1
                            
                            quarantine_prob = np.random.random()
                            if quarantine_prob >= self.gamma:
                                self.new_epidemic_state[node] = 'I'
                            else:
                                self.new_epidemic_state[node] = 'Q'

            
            if self.new_epidemic_state[node] == 'V':
                
                if self.time_immune[node] >= self.immunation_time[node]:
                    
                    self.time_immune[node] = 0
                    self.new_epidemic_state[node] = 'S'
                
                else:

                    self.time_immune[node] += 1
            
        elif self.epidemic_state[node] == 'I':
            
            if self.inf_time_list[node] >= self.inf_period_list[node]:
                
                death_prob = np.random.random()
                
                if self.ever_vaccinated[node] == 1:
                    
                    if death_prob < self.vacc_death_prob:
                        self.new_epidemic_state[node] = 'D'
                        self.time_immune[node] = 0
                        self.inf_time_list[node] = 0
                    else:
                        self.new_epidemic_state[node] = 'R'
                        self.time_immune[node] = 0
                        self.inf_time_list[node] = 0
                else:
                    
                    if death_prob < self.kappa:
                        self.new_epidemic_state[node] = 'D'
                        self.time_immune[node] = 0
                        self.inf_time_list[node] = 0
                    else:
                        self.new_epidemic_state[node] = 'R'
                        self.time_immune[node] = 0
                        self.inf_time_list[node] = 0
                        
            else:
                self.inf_time_list[node] += 1
        
        elif self.epidemic_state[node] == 'Q':
            
            if self.inf_time_list[node] >= self.inf_period_list[node]:
                
                death_prob = np.random.random()
                
                if self.ever_vaccinated[node] == 1:
                    
                    if death_prob < self.vacc_death_prob:
                        self.new_epidemic_state[node] = 'D'
                        self.time_immune[node] = 0
                        self.inf_time_list[node] = 0
                    else:
                        self.new_epidemic_state[node] = 'R'
                        self.time_immune[node] = 0
                        self.inf_time_list[node] = 0
                        
                else:
                    
                    if death_prob < self.kappa:
                        self.new_epidemic_state[node] = 'D'
                        self.time_immune[node] = 0
                        self.inf_time_list[node] = 0
                    else:
                        self.new_epidemic_state[node] = 'R'
                        self.time_immune[node] = 0
                        self.inf_time_list[node] = 0
                    
            else:
                self.inf_time_list[node] += 1
            
        elif self.epidemic_state[node] == 'R':

            if self.time_immune[node] >= self.immunation_time[node]:
                
                if self.infected_while_vacc[node] == 1:
                    self.new_epidemic_state[node] = 'V'
                    self.time_immune[node] = 0
                    self.infected_while_vacc[node] = 0
                    
                else:
                    self.new_epidemic_state[node] = 'S'
                    self.time_immune[node] = 0

            else:
                self.time_immune[node] += 1
        
        # elif self.epidemic_state[node] == 'D':
        #     self.new_epidemic_state[node] = 'D'
                    
    def check_opinion(self, node):
        
        if self.new_epidemic_state[node] != 'D':

            neighbours = np.array(self.opinion_network[node])
            l_neighbours = neighbours[self.new_epidemic_state[neighbours] != 'D']
            
            n_count = l_neighbours.shape[0]
        
            if n_count > 0:
                
                current_state = self.opinion_state[node]
                
                m_i = np.sum(self.opinion_state[l_neighbours])/n_count
                
                if current_state == -1:
                     
                    S_i = m_i * (1 - self.stubbornness[node]) + self.stubbornness[node] * current_state
                           
                    if S_i > 0:
                        
                        opinion_change_prob = np.random.random()
                        
                        if opinion_change_prob < np.abs(S_i):
                            self.new_opinion_state[node] = 0
     
                elif current_state == 1:
                    
                    S_i = m_i * (1 - self.stubbornness[node]) + self.stubbornness[node] * current_state
                           
                    if S_i < 0:
                        
                        opinion_change_prob = np.random.random()
                        
                        if opinion_change_prob < np.abs(S_i):
                            self.new_opinion_state[node] = 0                
              
                elif current_state == 0:
                    
                    i_neighbours = l_neighbours[self.new_epidemic_state[l_neighbours] == 'I']
                    i_count = i_neighbours.shape[0]
                    
                    S_i = m_i + self.omega * i_count/n_count
                            
                    if S_i < 0:
                        
                        opinion_change_prob = np.random.random()
                        
                        if opinion_change_prob < np.abs(S_i)/(1+self.omega):
                            self.new_opinion_state[node] = -1
                        
                    elif S_i > 0:
                        
                        opinion_change_prob = np.random.random()
                        
                        if opinion_change_prob < np.abs(S_i)/(1+self.omega):
                            
                            self.new_opinion_state[node] = 1                
                            
    def do_one_sim_step(self, step):

        nodes_check_order = np.random.permutation(self.nodes_list)
        
        self.new_epidemic_state = copy.deepcopy(self.epidemic_state)
        
        for node in nodes_check_order:
            
            self.check_epidemic(node)

        for _ in range(self.v_step):
            
            self.new_opinion_state = copy.deepcopy(self.opinion_state)
            
            for node in nodes_check_order:
                
                self.check_opinion(node)
        
            self.opinion_state = self.new_opinion_state
            
            # nodes_check_order = np.random.permutation(self.nodes_list)
        
        self.epidemic_state = self.new_epidemic_state
                    
        if step >= self.vacc_start:
        
            potential_vacc_nodes = np.logical_and(np.logical_and(self.epidemic_state == 'S', self.opinion_state == 1), self.ever_vaccinated == 0)
            potential_vacc_nodes_number = int(np.sum(potential_vacc_nodes))
            
            if potential_vacc_nodes_number > 0:
                
                if potential_vacc_nodes_number > self.daily_vacc_limit:
                
                    temp_state = np.full(potential_vacc_nodes_number, 'S')
            
                    vacc_nodes = np.random.choice(potential_vacc_nodes_number, self.daily_vacc_limit, replace=False)
                    temp_state[vacc_nodes] = 'V'
                    
                    self.epidemic_state[potential_vacc_nodes] = temp_state
                    self.ever_vaccinated[self.epidemic_state == 'V'] = 1
                    
                else:
                    self.epidemic_state[potential_vacc_nodes] = np.full(potential_vacc_nodes_number, 'V')
                    self.ever_vaccinated[self.epidemic_state == 'V'] = 1