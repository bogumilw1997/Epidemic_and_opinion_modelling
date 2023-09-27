
# Epidemic and opinion modelling

Python code for the numerical simulation used for my master thesis "Modelling epidemic spreading using social attitude to vaccines".

For epidemic and opinion spreading I'm using complex network topology, handled by the `networkx` package.
## Contents

* `stationary` folder contains code used for simulations done on stationary networks (conections between nodes doesn't change with time)    
    * `basic` folder contains the baseline version of the model
    * `stubbornness` folder contains the baseline model with added stubbornness coefficient
    * `vacc_limit` folder contains the model with stubbornness and added daily vaccination limit
    * `vanishing_immunity` folder contains the model with daily vaccination limit and added vanishing immunity afeter infection or vaccination
- `temporal` folder contains code used for simulations done on temporal networks (conections between nodes change with time, here I'm using the Activity Driven Network model)
   * `basic` folder contains the baseline version of the model
   * `lower_inf_act` folder contains the baseline version of the model with added lower activity for infected nodes
## Parameters

In each folder, file `params.json` controls the parameters of the simulation.

* `N` controls the number of nodes at both epidemic and opinion level
* `m` is the number of nodes added at each construction step of the BA network
* `T` is the number of Monte Carlo simulation steps 
* `p0_inf` is the probability that the node will be infected at T=0
* `p0_op_minus` is the probability that the node will havr -1 opinion at T=0
* `p0_op_plus` is the probability that the node will have +1 opinion at T=0
* `beta` is the epidemic spread rate
* `omega` is the recovery probability
* `mu` is the probability of returning to the S/V state
* `waning_time` is the time after which the vaccine stops protecting against the infection
* `vacc_death_prob` is the probability of the agent dying from infection after beeing vaccinated
* `v_step` is the amount of the opinion simulation steps per epidemic step
* `vacc_start` is the first simulation step at which agents can be vaccinated
* `vacc_limit` is teh maximum relative number of the agents  that can be vaccinated per 1 simulation step
* `realizations` is the number of the MC simulatuon realizations per 1 parameters set
