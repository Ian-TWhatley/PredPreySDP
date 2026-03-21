# PredPreySDP

This is a demonstration that simulates Stochastic Dynamic Programming (SDP) over a predator prey system, creating a best policy of harvesting prey from either 0 to 10.

The base model is given as the discrete-time ODE as follows:

$$ \begin{aligned} \xi N_{t+1} &= rN_t(1 - \frac {N_t}{K}) - a N_t P_t \\
 \xi P_{t+1} & = bP_tN_t - \mu P_t  \end{aligned} $$

In which each timestep has stochastic noise applied given by $\xi \sim \text{Lognormal}(0,\sigma^2)$.

Multiple simulations are run under this model to give the transition matrices of each action from harvesting 0 to 10. Then, after this, SDP is performed given a choice of different strategies.

Everything can be run using the UI seen below.

<img width="1918" height="1012" alt="app_overview" src="https://github.com/Ian-TWhatley/PredPreySDP/main/pics_vids/app_overview.png"/>

## How to Run
In order to run, you must use streamlit.

To run (in bash terminal):
```
streamlit run main.py
```
or try
```
python -m streamlit run main.py
```


## Important Note
If order to load the data, you need to pickle a file. To do this, simply run `run_SDP.py`, which will then create a .pkl file.
