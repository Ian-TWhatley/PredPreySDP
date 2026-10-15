# PredPreySDP

This is a demonstration that simulates Stochastic Dynamic Programming (SDP) over a predator prey system, creating a best policy of harvesting prey.

The base model is given as the follow system of ODEs:

$$
\begin{align*}
    \frac{dN}{dt} &= rN - a N P \\
    \frac{dP}{dt} &= bNP - d 
\end{align*}
$$

We then use Heun's method to discretize the ODE series:

$$
\begin{align*}
    f(N_t,P_t)&:=\frac{dN_t}{dt} \\
    (N',P') &= (N_t + f(N_t,P_t), P_t + f(N_t,P_t))  \\
    N_{t+1} &= \xi \bigg[\frac1 2 \big(f(N',P')+f(N_t,P_t)\big)\bigg]
\end{align*}
$$

In which each timestep has stochastic noise applied given by $\xi \sim \text{Lognormal}(0,\sigma^2)$.

We also have an action space $A$ where each $a \in A$ is the number of prey harvested. In this case, our action space depends on the given state since we cannot harvest more prey than currently exist. We denote $A_n$ as the action space where we have $n$ prey.

$$
\mathcal A = \bigcup_{\substack {i=0 \\ \text{step}= \Delta}}^{N_{max}}A_i
$$

For example, if have $\Delta=0.5$, $A_2=[0,0.5,1,1.5,2]$

From $\mathcal A$, we then need to find our probability distribution. We can break this up into probabilities based on initial values where we have random vector 

$$
V_{(N_t,P_t)} \sim\text{lognormal}\big( [N_{t+1},\ P_{t+1}]^\top, \sigma^2 I_2\big)
$$

We then must discretize our vectors to find a a transition matrix $T_a$ composed of $p_{i,j}:=P(S_{j,t+1}-a|S_{i,t})$ over our finite states. In order to this we apply linear interpolation. Therefore:

$$
\mathcal T_a  = \begin{bmatrix}
p_{(0,0),(0,0)}  & \cdots & p_{(0,0),(N_{\max},P_{\max})} \\
\vdots    & \ddots & \vdots \\
p_{(N_{\max},P_{max}),(0,0)}  & \cdots & p_{(N_{\max},P_{\max}),(N_{\max},P_{\max})}
\end{bmatrix}
$$

<img width="1918" height="1012" alt="app_overview" src="https://github.com/Ian-TWhatley/PredPreySDP/blob/main/pics_vids/app_overview.png"/>

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
