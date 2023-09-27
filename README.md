
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
