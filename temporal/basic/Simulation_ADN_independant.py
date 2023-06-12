from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import copy
from collections import Counter
import seaborn as sns
from scipy.optimize import curve_fit
import math
import random
import sys
import powerlaw

class Simulation():
    
    opinion_states = np.array([-1, 0, 1])
    epidemic_states = np.array(['S', 'I', 'R', 'V'])
    
    def power_dist(self, r, xmin, w):
        return xmin * (1-r) ** (-1/(w-1))
    
    def __init__(self, N, m, beta, omega, gamma, mu, phi, kappa, v_step, eta, epsilon, w):
        
        self.N = N
        self.m = m
        self.w = w
        
        self.v_step = v_step
        
        self.nodes_list = np.arange(N) 
        
        self.beta = beta
        self.omega = omega
        self.gamma = gamma
        self.mu = mu
        self.phi = phi
        self.kappa = kappa
        
        #self.epidemic_activity = self.power_dist(np.random.uniform(size = N), epsilon, w)
        self.epidemic_activity = powerlaw.Power_Law(xmin=epsilon, parameters=[w]).generate_random(N)
        self.epidemic_activity_rate = np.minimum(self.epidemic_activity * eta, np.ones_like(self.epidemic_activity))
        
        #self.opinion_activity = self.power_dist(np.random.uniform(size = N), epsilon, w)
        self.opinion_activity = powerlaw.Power_Law(xmin=epsilon, parameters=[w]).generate_random(N)
        self.opinion_activity_rate = np.minimum(self.opinion_activity * eta, np.ones_like(self.opinion_activity))
        
        self.ever_vaccinated = np.zeros(self.N)
        self.ever_infected = np.zeros(self.N)
        # self.inf_while_vacc = np.zeros(self.N)
        
    def init_states(self, p0_inf, p0_op_minus, p0_op_plus):
        
        self.epidemic_state = np.random.choice(self.epidemic_states[:2], self.N, p=[1- p0_inf, p0_inf])
        self.opinion_state =  np.random.choice(self.opinion_states, self.N, p=[p0_op_minus, 1 - p0_op_plus - p0_op_minus,p0_op_plus])
        
        #self.new_epidemic_state = copy.deepcopy(self.epidemic_state)
        #self.new_opinion_state = copy.deepcopy(self.opinion_state)
        
        self.ever_infected[self.epidemic_state == 'I'] = 1  
       
    def vaccinate_nodes(self):
        
        potential_vacc_nodes = np.logical_and(self.epidemic_state == 'S', self.opinion_state == 1)
        potential_vacc_nodes_number = int(np.sum(potential_vacc_nodes))
        
        if potential_vacc_nodes_number > 0:
            
            self.epidemic_state[potential_vacc_nodes] = np.random.choice(["S", "V"], potential_vacc_nodes_number, p=[1- self.phi, self.phi])
            self.ever_vaccinated[self.epidemic_state == 'V'] = 1
        
    def get_inf_number(self):
        return np.sum(self.epidemic_state == "I")/self.N
    
    def get_vacc_number(self):
        return np.sum(self.epidemic_state == "V")/self.N
    
    def get_susc_number(self):
        return np.sum(self.epidemic_state == "S")/self.N

    def get_quar_number(self):
        return np.sum(self.epidemic_state == "Q")/self.N
    
    def get_recov_number(self):
        return np.sum(self.epidemic_state == "R")/self.N
    
    def get_dead_number(self):
        return np.sum(self.epidemic_state == "D")/self.N
    
    def get_opinion_plus(self):
        return np.sum(self.opinion_state == 1)/self.N
    
    def get_opinion_minus(self):
        return np.sum(self.opinion_state == -1)/self.N
    
    def get_opinion_zero(self):
        return np.sum(self.opinion_state == 0)/self.N
    
    def get_total_inf_number(self):
        return np.mean(self.ever_infected)
    
    def get_total_vacc_number(self):
        return np.mean(self.ever_vaccinated)
    
    def get_mean_opinion(self):
        return np.mean(self.opinion_state)
    
    def get_epidemic_graph(self):
        return self.epidemic_network
    
    def get_opinion_graph(self):
        return self.opinion_network
    
    def get_epidemic_degrees(self):
        return np.array([self.epidemic_network.degree[_] for _ in range(self.N)])
    
    def get_opinion_degrees(self):
        return np.array([self.opinion_network.degree[_] for _ in range(self.N)])
    
    def get_total_immune_number(self):
        return self.get_vacc_number() + self.get_recov_number() + np.sum(np.logical_and(self.epidemic_state == 'I', self.infected_while_vacc == 1))/self.N + np.sum(np.logical_and(self.epidemic_state == 'Q', self.infected_while_vacc == 1))/self.N
    
    def make_opinion_histogram(self, ax):    
        states = ["-1", "0", "+1"]
        values = [np.sum(self.opinion_state == -1)/self.N, np.sum(self.opinion_state == 0)/self.N, np.sum(self.opinion_state == 1)/self.N]
        
        ax.bar(states, values,color=['lightcoral', 'lightskyblue', 'lightgreen'])
        ax.grid(visible= True, axis="y")
        
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
    
    def check_epidemic_topology(self):
        
        self.epidemic_network = nx.empty_graph(self.N, nx.Graph)
        
        if self.get_inf_number() > 0.0:
            
            probs = np.random.rand(self.N)
        
            active_nodes = self.nodes_list[probs <= self.epidemic_activity_rate]
            
            for active_node in active_nodes:
                    
                nodes_to_connect = np.random.choice(self.nodes_list, self.m, replace=False)
                
                while(np.isin(active_node, nodes_to_connect, assume_unique=True)):
                    nodes_to_connect = np.random.choice(self.nodes_list, self.m, replace=False)
                    
                #nodes_to_connect = nodes_to_connect[self.epidemic_state[nodes_to_connect] == "I"]
                
                if nodes_to_connect.shape[0] > 0:
                    zz = zip(np.full(nodes_to_connect.shape[0], active_node), nodes_to_connect)
                    
                    self.epidemic_network.add_edges_from(zz)
            
    def check_opinion_topology(self):
        
        self.opinion_network = nx.empty_graph(self.N, nx.Graph)
        
        if np.sum(self.opinion_state != 0) > 0:
            
            probs = np.random.rand(self.N)
        
            active_nodes = self.nodes_list[probs <= self.opinion_activity_rate]
            
            for active_node in active_nodes:
                    
                nodes_to_connect = np.random.choice(self.nodes_list, self.m, replace=False)
                
                while(np.isin(active_node, nodes_to_connect, assume_unique=True)):
                    nodes_to_connect = np.random.choice(self.nodes_list, self.m, replace=False)
                    
                #nodes_to_connect = nodes_to_connect[self.opinion_state[nodes_to_connect] != 0]
                
                if nodes_to_connect.shape[0] > 0:
                    zz = zip(np.full(nodes_to_connect.shape[0], active_node), nodes_to_connect)
                    
                    self.opinion_network.add_edges_from(zz)
       
    def check_topology(self):
        
        self.check_epidemic_topology()
        self.check_opinion_topology()
                
    def check_epidemic(self, node):
        
        if self.epidemic_state[node] == 'S':
            
            neighbours = np.array(self.epidemic_network[node])
            
            if neighbours.shape[0] > 0:
                    
                    i_neighbours = neighbours[self.epidemic_state[neighbours] == 'I']
                    
                    i_count = i_neighbours.shape[0]
                    
                    if i_count > 0:
                        
                        infection_prob = np.random.random_sample(size = i_count)
                        
                        if all(i_p >= self.beta for i_p in infection_prob):
                            return "S"
                        else:
                            self.ever_infected[node] = 1
                            return "I"
                    else:
                        return "S"
            else:
                return "S"
                
        elif self.epidemic_state[node] == 'V':
            
            neighbours = np.array(self.epidemic_network[node])
            
            if neighbours.shape[0] > 0:
                    
                    i_neighbours = neighbours[self.epidemic_state[neighbours] == 'I']
                    
                    i_count = i_neighbours.shape[0]
                    
                    if i_count > 0:
                        
                        infection_prob = np.random.random_sample(size = i_count)
                        
                        if all(i_p >= (self.beta * (1 - self.omega))  for i_p in infection_prob):
                            return "V"
                        else:
                            #self.inf_while_vacc[node] = 1
                            self.ever_infected[node] = 1
                            return "I"
                    else:
                        return "V"
            else:
                return "V"
                
        elif self.epidemic_state[node] == 'I':
            
            rec_prob = np.random.random()
            
            if rec_prob < self.gamma:
                return "R" 
            else:
                return "I"

        elif self.epidemic_state[node] == 'R':
            
            comeback_prob = np.random.random()
                
            if comeback_prob < self.mu:
                
                if self.ever_vaccinated[node] == 1:
                    return 'V'
                    
                else:
                    return "S"
            else:
                return "R"
                    
    def check_opinion(self, node):

        current_opinion = self.opinion_state[node]
        
        if current_opinion != 0:
            
            prob = np.random.random()
            
            if prob < self.kappa:    
                return 0
            else:
                return current_opinion
        else:
        
            neighbours = np.array(self.opinion_network[node])
            
            if neighbours.shape[0] > 0:
                
                active_neighbours = neighbours[self.opinion_state[neighbours] != 0]
                
                if active_neighbours.shape[0] > 0:
                    
                    plus_neighbours =  np.sum(self.opinion_state[active_neighbours] == 1)
                    minus_neighbours =  np.sum(self.opinion_state[active_neighbours] == -1)
                    
                    if plus_neighbours > 0:
                        p_prob = np.random.random_sample(size = plus_neighbours)
                        p = np.sum(p_prob < self.omega)
                    else:
                        p = 0
                    
                    if minus_neighbours > 0:
                        m_prob = np.random.random_sample(size = minus_neighbours)
                        m = np.sum(m_prob < (1 - self.omega))
                    else:
                        m = 0
                    
                    if p > m:
                        return 1
                    elif p < m:
                        return -1
                    else:
                        return current_opinion
                else:
                    return current_opinion
            else:
                return current_opinion
                              
    def do_one_sim_step(self):
        
        self.inf_number = self.get_inf_number()
        
        self.check_topology()
        
        #self.new_epidemic_state = copy.deepcopy(self.epidemic_state)
        
        e = map(self.check_epidemic, self.nodes_list)
        
        self.epidemic_state = np.array(list(e))
        
        #self.new_epidemic_state = np.array(list(e))
        
        # for node in self.nodes_list:
            
        #     self.check_epidemic(node)
            
        for _ in range(self.v_step):
            
            #self.new_opinion_state = copy.deepcopy(self.opinion_state)
            
            # for node in self.nodes_list:
                
            #     self.check_opinion(node)

            o = map(self.check_opinion, self.nodes_list)
            
            self.opinion_state = np.array(list(o))
        
        #self.epidemic_state = self.new_epidemic_state
                    
        self.vaccinate_nodes()