import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats

class PredatorPreySystem:
    """
    Discrete predator-prey ODE simulation with stochastic noise and a stochastic dynamic programming component

    Attributes:
        extinct (dict): Dictionary to store extinction times for managed and unmanaged populations
        ext_mats (dict): Matrices to store extinction counts and times
        cull_mat (np.array): The amount of culling applied to each state given intial populations
        N_list (dict): Lists to store prey population trajectories for managed and unmanaged populations
        P_list (dict): Lists to store predator population trajectories for managed and unmanaged populations
        S_mat (np.array): State matrix containing all possible population states
        num_states (int): Number of states in the state matrix
        T_mats (dict): Transition matrices for different culling strategies
        N_max (int): Maximum prey population
        P_max (int): Maximum predator population
    """
    def __init__(self, N_max = 40, P_max=30):
        """
        Initialize an instance
        
        :param N_max: The maximum prey population
        :param P_max: The maximum predator population
        """
        self.S_mat = np.array([[n,p] for n in range(N_max+1) for p in range(P_max+1)], dtype=float)
        self.num_states = self.S_mat.shape[0]
        self.T_mats = {}
        self.N_max = N_max
        self.P_max = P_max
        pass

    def run(
            self,
            d=2,
            a=0.01,
            K=100,
            r=1.3,
            b=0.08,
            mu=0.1,
            sigma=0.05,
            matrix_name:str = None,
            cull:str = 'none',
            N_cull:int = 5,
            P_cull:int = 5,
        ) -> None:
        """
        Run simulations to build transition matrix

        :param d: Extinction threshold
        :param a: Predator hunt success rate
        :param K: Prey carrying capacity
        :param r: Prey growth rate
        :param b: Predator birth rate, based on prey abundance
        :param mu: Predator mortality rate
        :param sigma: Environmental stochasticity
        :param matrix_name: Name of the transition matrix, suggested T Matrix, T N Matrix 1, etc.
        :param time: Maximum time steps per simulation
        :param sims: Number of simulations per state
        :param cull: Type of culling ('prey', 'predator', 'both', or 'none')
        :param N_cull: Number of prey to cull if culling prey
        :param P_cull: Number of predators to cull if culling predators
        """
        mat = np.zeros((self.num_states,self.num_states),dtype=float) # Set matrix, may be anything
        for i in range(self.num_states):
            # Initial Values
            N = self.S_mat[i,0]
            P = self.S_mat[i,1]

            # Deterministic averages
            N_next = r*N*(1 - N / K) - a*N*P
            P_next = b*P*N - mu*P

            if cull == 'prey':
                N_next = N_next - N_cull

            if cull == 'predator':
                P_next = P_next - P_cull

            if cull == 'both':
                N_next = N_next - N_cull
                P_next = P_next - P_cull
            
            # Ensure base values are positive for log-space math
            N_next = max(N_next, 1e-9)
            P_next = max(P_next, 1e-9)

            # Possible N values and probabilities given lognormal noise
            n_vals = np.arange(self.N_max + 1)
            n_probs = stats.lognorm.cdf(n_vals + 0.5, s=sigma, scale=N_next) - stats.lognorm.cdf(n_vals - 0.5, s=sigma, scale=N_next)
            
            # Possible P values and probabilities given lognormal noise
            p_vals = np.arange(self.P_max + 1)
            p_probs = stats.lognorm.cdf(p_vals + 0.5, s=sigma, scale=P_next) - stats.lognorm.cdf(p_vals - 0.5, s=sigma, scale=P_next)

            # Boundary handling (Extinction and Max Capacity)
            # Prob(0) includes everything below 0.5
            n_probs[0] = stats.lognorm.cdf(d-0.5, s=sigma, scale=N_next)
            n_probs[1:d] = 0 # Set probabilities for states between 0 and d to 0
            p_probs[0] = stats.lognorm.cdf(d-0.5, s=sigma, scale=P_next)
            p_probs[1:d] = 0 # Set probabilities for states between 0 and d to 0
            
            # Prob(Max) includes everything above Max - 0.5
            n_probs[-1] = 1.0 - stats.lognorm.cdf(self.N_max - 0.5, s=sigma, scale=N_next)
            p_probs[-1] = 1.0 - stats.lognorm.cdf(self.P_max - 0.5, s=sigma, scale=P_next)

            # Fill the transition matrix row using outer product
            # This gives the joint probability P(N=n and P=p) assuming independence of noise
            joint_probs = np.outer(n_probs, p_probs).flatten()
            mat[i, :] = joint_probs
            self.T_mats[matrix_name] = mat
    
    def reset_lists(self) -> None:
        """
        Reset the lists used to store population trajectories and extinction times
        """
        self.extinct = {'managed':[], 'unmanaged':[]}
        self.ext_mats = None
        self.cull_mat = None
        self.N_list = {'managed':[], 'unmanaged':[]}
        self.P_list = {'managed':[], 'unmanaged':[]}
        self.cull_total = 0
    
class SDP(PredatorPreySystem):
    def __init__(self, system: PredatorPreySystem):
        # Inherit from PredatorPreySystem
        self.N_max = system.N_max
        self.P_max = system.P_max
        self.T_mats = system.T_mats
        self.S_mat = system.S_mat
        self.num_states = system.num_states
        # New attributes for SDP
        self.extinct = {'managed':[], 'unmanaged':[]}
        self.ext_mats = None
        self.cull_mat = None
        self.N_list = {'managed':[], 'unmanaged':[]}
        self.P_list = {'managed':[], 'unmanaged':[]}

    def _optimization_algo(
            self,
            backward,
            terminal_val,
            R_mat,
            t_max,
            num_actions,
            disc_factor,
            t_threshold,
            tol,
            policy_iter,
            value_iter,
        ) -> None:

        # Backward iteration to find optimal policy
        self.A_mat = np.zeros((self.num_states, t_max), dtype=int)
        # Backward iteration to find optimal policy
        if backward == True:
            V = terminal_val # terminal value
            for t in range(t_max):
                V_old = V.copy()
                Q = np.zeros((self.num_states, num_actions))
                for h in range(num_actions):
                    Q[:, h] = self.R_mat[:,h] + disc_factor*list(self.T_mats.items())[h][1] @ V
                    
                V = np.max(Q, axis=1)
                self.A_mat[:, t_max - t -1] = np.argmax(Q,axis=1)
                if t > t_threshold:
                    if value_iter:
                        if np.linalg.norm(V - V_old) <= tol:
                            self.stop = t
                            self.A_mat[:, 0] = np.argmax(Q,axis=1)
                            break
                    if policy_iter:
                        if np.linalg.norm(self.A_mat[:,t_max -t -1] - self.A_mat[:,t_max - t]) <= tol:
                            self.stop = t
                            self.A_mat[:, 0] = np.argmax(Q,axis=1)
                            break
            
            # Save Policy Matrix
            self.P_mat = np.zeros((self.N_max, self.P_max))

            for i in range(self.num_states):
                n = int(self.S_mat[i, 0])
                p = int(self.S_mat[i, 1])
                self.P_mat[n-1, p-1] = self.A_mat[i, 0]

        # Policy iteration to find optimal policy
        if backward == False:
            V = np.zeros(self.num_states)
            for t in range(t_max):
                Q = np.zeros((self.num_states, num_actions))
                for h in range(num_actions):
                    Q[:, h] = self.R_mat[:,h] + disc_factor*list(self.T_mats.items())[h][1] @ V
                    
                V = np.max(Q, axis=1)
                self.A_mat[:, t] = np.argmax(Q,axis=1)
                if t > t_threshold:
                    if np.linalg.norm(self.A_mat[:,t] - self.A_mat[:,t-1]) <= tol:
                        self.stop = t
                        self.A_mat[:, -1] = np.argmax(Q,axis=1)
                        break
            
            # Save Policy Matrix
            self.P_mat = np.zeros((self.N_max, self.P_max))

            for i in range(self.num_states):
                n = int(self.S_mat[i, 0])
                p = int(self.S_mat[i, 1])
                self.P_mat[n-1, p-1] = self.A_mat[i, -1]

    
    def sdp_standard(
            self,
            t_max: int=100,
            tol:float = 0,
            t_threshold:float = 10,
            backward=True,
            value_iter = True,
            policy_iter = False,
            disc_factor:float = 1
        ):
        """
        Perform stochastic dynamic programming to find optimal culling strategy. This was taken from (3) Munch et al.
        Very interesting stuff considering everything. Only change was that I made the value multiplied instead. Better Results.
        I have to think about hte intution as to why this works.

        JODY: DISREGARD, CURRENTLY BROKEN

        :param t_max: Maximum time horizon
        """
        self.stop = None
        self.cull_total = 0
        num_actions = len(list(self.T_mats.items()))

        self.R_mat = np.zeros((self.num_states, num_actions))
        for s in range(self.num_states): # Over all possible combination of prey and predator population sizes, find reward for each action and pop
            N_i =  float(self.S_mat[s,0]) # Initialize population size
            P_i =  float(self.S_mat[s,1])
            for h in range(0,num_actions):
                if h > N_i: # Make sure that harvesting more is impossible
                    self.R_mat[s, h] = -np.inf
                    continue
                if N_i == 0:
                    self.R_mat[s, h] = 0
                    continue
                else:
                    self.R_mat[s, h] = N_i*P_i # Reward is number harvested, cannot harvest more than population size

        self._optimization_algo(
            backward,
            self.S_mat.prod(axis=1),
            self.R_mat,
            t_max,
            num_actions,
            disc_factor,
            t_threshold,
            tol,
            policy_iter,
            value_iter,
        )
    
    def sdp_bode(
            self, 
            t_max:int = 500,
            tol:float = 1e-8,
            t_threshold:float = 10,
            backward=True,
            policy_iter = False,
            value_iter = True,
            disc_factor:float = 1,
        ):
        """
        Perform stochastic dynamic programming to find optimal culling strategy

        JODY: DISREGARD

        :param t_max: Maximum time horizon
        """
        self.stop = None
        self.cull_total = 0
        num_actions = len(list(self.T_mats.items()))
        self.R_mat = np.zeros((self.num_states,num_actions))

        self._optimization_algo(
            backward,
            self.S_mat.prod(axis=1),
            self.R_mat,
            t_max,
            num_actions,
            disc_factor,
            t_threshold,
            tol,
            policy_iter,
            value_iter,
        )

    def sdp_dual_harv(
            self,
            h_max:int,
            c_max:int,
            t_max=250,
            disc_factor:float = 1,
            policy_iter = True,
            value_iter = True,
            tol=0,
            t_threshold=10,
            backward=True
        )-> None:
        '''
        Rewards harvesting the most prey over a large time horizon, with a discount factor.
        '''
        self.stop = None
        self.cull_total = 0
        self.actions = [[h,c] for h in range(0,h_max+1) for c in range(0,c_max+1)]
        num_actions = len(self.actions)

        self.R_mat = np.zeros((self.num_states, num_actions))
        for s in range(self.num_states): # Over all possible combination of prey and predator population sizes, find reward for each action and pop
            N_i = float(self.S_mat[s,0]) # Initialize population size
            P_i = float(self.S_mat[s,1])
            for a in range(num_actions):
                h = self.actions[a][0]
                c = self.actions[a][1]
                if h > N_i:
                    self.R_mat[s, a] = -np.inf
                    continue
                if c > P_i:
                    self.R_mat[s, a] = -np.inf
                    continue
                self.R_mat[s,a] = h + c

        self._optimization_algo(
            backward,
            self.S_mat.sum(axis=1),
            self.R_mat,
            t_max,
            num_actions,
            disc_factor,
            t_threshold,
            tol,
            policy_iter,
            value_iter,
        )

    def sdp_harv(
            self,
            t_max:int=250,
            disc_factor:float = 1,
            policy_iter = False,
            value_iter = True,
            tol=0,
            t_threshold=10,
            backward=True
        )-> None:
        '''
        Rewards harvesting the most prey over a large time horizon, with a discount factor.
        '''
        self.stop = None
        self.cull_total = 0
        num_actions = len(list(self.T_mats.items()))

        self.R_mat = np.zeros((self.num_states, num_actions))
        for s in range(self.num_states): # Over all possible combination of prey and predator population sizes, find reward for each action and pop
            N_i =  float(self.S_mat[s,0]) # Initialize population size
            for h in range(0,num_actions):
                if h > N_i:
                    self.R_mat[s, h] = -np.inf
                    continue
                self.R_mat[s, h] = min(h,N_i) # Reward is number harvested 

        self.A_mat = np.zeros((self.num_states, t_max), dtype=int)
        
        self._optimization_algo(
            backward,
            self.S_mat.sum(axis=1),
            self.R_mat,
            t_max,
            num_actions,
            disc_factor,
            t_threshold,
            tol,
            policy_iter,
            value_iter,
        )

    def model_pop(
            self,
            cull:bool = True,
            N_0 = None,
            P_0 = None,
            d=2,
            a=0.01,
            K=100,
            r=1.3,
            b=0.08,
            mu=0.1,
            sigma=0.05,
            time=200,
        ) -> None:
        """
        Simulate population dynamics given initial populations and parameters

        :param cull: Whether to apply culling based on the optimal strategy
        :param N_0: Initial prey population
        :param P_0: Initial predator population
        :param d: Extinction threshold
        :param a: Predator hunt success rate
        :param K: Prey carrying capacity
        :param r: Prey growth rate
        :param b: Predator birth rate, based on prey abundance
        :param mu: Predator mortality rate
        :param sigma: Environmental stochasticity
        :param time: Maximum time steps per simulation
        """
        if self.ext_mats == None:
            self.ext_mats = {'managed': [np.zeros((self.N_max+1,self.P_max+1))
                                        , np.zeros((self.N_max+1,self.P_max+1))], 
                            'unmanaged':[np.zeros((self.N_max+1,self.P_max+1))
                                        , np.zeros((self.N_max+1,self.P_max+1))]}
        if self.cull_mat is None:
            self.cull_mat = np.zeros((self.N_max+1,self.P_max+1))
        if N_0 == None and P_0 == None:
            N_0 = random.uniform(d+1, self.N_max)
            P_0 = random.uniform(d+1, self.P_max)

        N = [N_0]
        P = [P_0]
        for t in range(1, time):
            # stochasticity
            Z_N = random.lognormvariate(0, sigma)
            Z_P = random.lognormvariate(0, sigma)

            # prey update
            N_next = r*N[t-1]*(1 - N[t-1] / K) - a*N[t-1]*P[t-1]

            # predator update
            P_next = b*P[t-1]*N[t-1] - mu*P[t-1]

            # check if we need to cull
            if cull:
                if N[t-1] > self.N_max and P[t-1] > self.P_max:
                    N_next += -self.P_mat[self.N_max - 1, self.P_max - 1]
                    self.cull_total += self.P_mat[self.N_max - 1, self.P_max - 1]
                    self.cull_mat[N_0, P_0] += self.P_mat[self.N_max - 1, self.P_max - 1]
                if N[t-1] > self.N_max:
                    N_next += -self.P_mat[self.N_max - 1, round(P[t-1])-1]
                    self.cull_total += self.P_mat[self.N_max - 1, round(P[t-1])-1]
                    self.cull_mat[N_0, P_0] += self.P_mat[self.N_max - 1, round(P[t-1])-1]
                if P[t-1] > self.P_max:
                    N_next += -self.P_mat[round(N[t-1])-1, self.P_max-1]
                    self.cull_total += self.P_mat[round(N[t-1])-1, self.P_max-1]
                    self.cull_mat[N_0, P_0] += self.P_mat[round(N[t-1])-1, self.P_max-1]
                else:
                    N_next += -self.P_mat[round(N[t-1])-1, round(P[t-1])-1]
                    self.cull_total += self.P_mat[round(N[t-1])-1, round(P[t-1])-1]
                    self.cull_mat[N_0, P_0] += self.P_mat[round(N[t-1])-1, round(P[t-1])-1]

            N.append(round(Z_N * N_next))
            P.append(round(Z_P * P_next))

            # extinction check
            if N[t] < d or P[t] < d:
                if cull:
                    self.extinct['managed'].append(len(N))
                    self.N_list['managed'].append(N)
                    self.P_list['managed'].append(P)
                    self.ext_mats['managed'][0][N_0,P_0] += 1
                    self.ext_mats['managed'][1][N_0,P_0] += t
                    break
                else:
                    self.extinct['unmanaged'].append(len(N))
                    self.N_list['unmanaged'].append(N)
                    self.P_list['unmanaged'].append(P)
                    self.ext_mats['unmanaged'][0][N_0,P_0] += 1
                    self.ext_mats['unmanaged'][1][N_0,P_0] += t
                    break
        if cull:
            self.N_list['managed'].append(N)
            self.P_list['managed'].append(P)
        else:
            self.N_list['unmanaged'].append(N)
            self.P_list['unmanaged'].append(P)

    def plot(self, type:list,i=None) -> plt.Figure:
        """
        Plot population time series

        :param type: 'managed' or 'unmanaged' population
        :param i: Index of the population trajectory to plot, defaults to the last one
        """
        if i is None:
            i = len(self.N_list[type]) - 1

        fig, ax = plt.subplots()

        ax.plot(self.N_list[type][i], label='prey')
        ax.plot(self.P_list[type][i], label='predator')

        ax.legend()

        return fig

    def plot_sdp(self)-> plt.Figure:
        '''
        Plot an optimal culling strategy

        Actions:
        0 - No cull
        1 - Cull predator
        2 - Cull prey
        '''

        if self.P_mat is None:
            raise ValueError("Policy matrix not set. Please run methods sdp or sdp_economic first.")

        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.pcolor(np.arange(self.N_max+1), np.arange(self.P_max+1), self.P_mat.T, shading='auto')
        fig.colorbar(c, ax = ax, label="Action")
        ax.set_xlabel("N")
        ax.set_ylabel("P")
        ax.set_title("Optimal Action Map")
        return fig

    def plot_dual_sdp(self, backwards: bool = True) -> plt.Figure:
        '''
        Plot an optimal culling strategy
        '''
        # Extract optimal actions for each state
        if backwards == True:
            optimal_actions = self.A_mat[:, 0] # Get the optimal action for each state at t=0
        if backwards == False:
            optimal_actions = self.A_mat[:, -1] # Get the optimal action for each state at the last iteration

        # Create matrices for prey and predator harvests
        prey_harvest_mat = np.zeros((self.N_max, self.P_max))
        pred_harvest_mat = np.zeros((self.N_max, self.P_max))

        for i in range(self.num_states):
            n = int(self.S_mat[i, 0]) - 1  # Adjust for 0-based indexing
            p = int(self.S_mat[i, 1]) - 1
            action_idx = optimal_actions[i]
            h, c = self.actions[action_idx]
            prey_harvest_mat[n, p] = h
            pred_harvest_mat[n, p] = c
        
        # Plot prey and predator harvests side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Prey harvest
        c1 = ax1.pcolor(np.arange(self.N_max+1), np.arange(self.P_max+1), prey_harvest_mat.T, shading='auto', vmin=0, vmax=prey_harvest_mat.max())
        fig.colorbar(c1, ax=ax1, label="Prey Harvested")
        ax1.set_xlabel("N")
        ax1.set_ylabel("P")
        ax1.set_title("Optimal Prey Harvest Map")

        # Predator harvest
        c2 = ax2.pcolor(np.arange(self.N_max+1), np.arange(self.P_max+1), pred_harvest_mat.T, shading='auto', vmin=0, vmax=pred_harvest_mat.max())
        fig.colorbar(c2, ax=ax2, label="Predators Harvested")
        ax2.set_xlabel("N")
        ax2.set_ylabel("P")
        ax2.set_title("Optimal Predator Harvest Map")

        return fig
    
    def _plot_extinction_matrices_percents(self) -> plt.Figure:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        # Unmanaged extinction matrix
        c1 = ax1.pcolor(np.arange(self.system.N_max+1), np.arange(self.system.P_max+1), self.ext_mats['unmanaged'][0].T/100, shading='auto',)
        fig.colorbar(c1, ax=ax1, label="Extinction %")
        ax1.set_xlabel("N_0")
        ax1.set_ylabel("P_0")
        ax1.set_title("Unmanaged Extinction Matrix")

        # Managed extinction matrix
        c2 = ax2.pcolor(np.arange(self.system.N_max+1), np.arange(self.system.P_max+1), self.ext_mats['managed'][0].T/100, shading='auto')
        fig.colorbar(c2, ax=ax2, label="Extinction %")
        ax2.set_xlabel("N_0")
        ax2.set_ylabel("P_0")
        ax2.set_title("Managed Extinction Matrix")

        c3 = ax3.pcolor(np.arange(self.system.N_max+1), np.arange(self.system.P_max+1), (self.ext_mats['unmanaged'][0]-self.ext_mats['managed'][0]).T/100, shading='auto')
        fig.colorbar(c3, ax=ax3, label="Extinction Difference %")
        ax3.set_xlabel("N_0")
        ax3.set_ylabel("P_0")
        ax3.set_title("Difference Matrix")

        plt.tight_layout()
        plt.show()

    def _plot_extinction_matrices_times(self) -> plt.Figure:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        # Unmanaged extinction matrix
        c1 = ax1.pcolor(np.arange(self.system.N_max+1), np.arange(self.system.P_max+1), self.ext_mats['unmanaged'][1].T/self.ext_mats['unmanaged'][0].T, shading='auto',)
        fig.colorbar(c1, ax=ax1, label="Average Extinction Time")
        ax1.set_xlabel("N_0")
        ax1.set_ylabel("P_0")
        ax1.set_title("Unmanaged Extinction Matrix")

        # Managed extinction matrix
        c2 = ax2.pcolor(np.arange(self.system.N_max+1), np.arange(self.system.P_max+1), 
                        self.ext_mats['managed'][1].T/self.ext_mats['managed'][0].T, shading='auto')
        fig.colorbar(c2, ax=ax2, label="Average Extinction Time")
        ax2.set_xlabel("N_0")
        ax2.set_ylabel("P_0")
        ax2.set_title("Managed Extinction Matrix")

        c3 = ax3.pcolor(np.arange(self.system.N_max+1), np.arange(self.system.P_max+1), 
                        (self.ext_mats['unmanaged'][1].T-self.ext_mats['managed'][1].T)/
                            (self.ext_mats['unmanaged'][0].T-self.ext_mats['managed'][0].T), shading='auto')
        fig.colorbar(c3, ax=ax3, label="Average Extinction Time Difference")
        ax3.set_xlabel("N_0")
        ax3.set_ylabel("P_0")
        ax3.set_title("Difference Matrix")

        plt.tight_layout()
        plt.show()