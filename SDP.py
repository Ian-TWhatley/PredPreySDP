import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Literal
from scipy import stats
from matplotlib.animation import FuncAnimation

class PredatorPreySystem:
    def __init__(self, N_max = None, P_max=None, box_size = 0.5):
        """
        Initialize an instance
        
        :param N_max: The maximum prey population
        :param P_max: The maximum predator population
        """
        self.N_max = N_max
        self.P_max = P_max
        self.box_size = box_size
        self.T_mats = {}
        self.cull_values = {'prey': [], 'predator': []}
        pass

    def run(
            self,
            death=0.1,
            a=0.01,
            r=0.02,
            b=0.05,
            d=0.11,
            sigma=0.05,
            K = 100,
            matrix_name:str = None,
            cull: Literal['prey', 'predator', 'none'] = 'none',
            N_cull:float = 5,
            P_cull:float = 5,
            logistic:bool = False,
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

        def _linear_interp(s, mu, sigma, delta, max):
            if s == 0:
                F_plus  = stats.lognorm.cdf(s + delta, s=sigma, scale=mu)
    
                G_plus  = stats.norm.cdf((np.log(s + delta) - np.log(mu)) / sigma, loc = sigma)
    
                return (
                (1+s/delta) * F_plus
                - (np.exp(np.log(mu) + 0.5 * sigma**2)/delta)
                * (G_plus)
                )
    
            if s == max:
                F       = stats.lognorm.cdf(s,         s=sigma, scale=mu)
                F_minus = stats.lognorm.cdf(s - delta, s=sigma, scale=mu)
    
                G       = stats.norm.cdf((np.log(s)         - np.log(mu)) / sigma, loc = sigma)
                G_minus = stats.norm.cdf((np.log(s - delta) - np.log(mu))   / sigma, loc = sigma)
    
                return (
                1 + (s/delta-1) * F_minus
                - (s/delta) * F
                + (np.exp(np.log(mu) + 0.5 * sigma**2)/delta)
                * (G - G_minus)
                )
    
            else:
                F_plus  = stats.lognorm.cdf(s + delta, s=sigma, scale=mu)
                F       = stats.lognorm.cdf(s,         s=sigma, scale=mu)
                F_minus = stats.lognorm.cdf(s - delta, s=sigma, scale=mu)
    
                G_plus  = stats.norm.cdf((np.log(s + delta) - np.log(mu)) / sigma, loc = sigma)
                G       = stats.norm.cdf((np.log(s)         - np.log(mu)) / sigma, loc = sigma)
                G_minus = stats.norm.cdf((np.log(s - delta) - np.log(mu))   / sigma, loc = sigma)
    
                return (
                    (1 + s / delta) * F_plus
                    - (2 * s / delta) * F
                    + (s / delta - 1) * F_minus
                    + (np.exp(np.log(mu) + 0.5 * sigma**2) / delta)
                    * (-G_minus + 2 * G - G_plus)
                )
        
        # Get parameters for system
        self.death, self.a, self.r, self.b, self.d, self.sigma, self.K = death, a, r, b, d, sigma, K

        # To pull from
        self.N_num = int((self.N_max/self.box_size)+1)
        self.P_num = int((self.P_max/self.box_size)+1)
        N_range = np.round(np.arange(0, self.N_max+self.box_size, self.box_size),3)
        P_range = np.round(np.arange(0, self.P_max+self.box_size, self.box_size),3)
            
        # Create state matrix and transition matrix
        self.S_mat = np.array([[n,p] for n in N_range
                               for p in P_range], dtype=float)
        self.num_states = self.S_mat.shape[0]

        mat = np.zeros((self.num_states,self.num_states),dtype=float) # Set matrix, may be anything
        
        for i in range(self.num_states):
            # Initial Values
            N = self.S_mat[i,0]
            P = self.S_mat[i,1]

            if N <= death: 
                n_probs = np.zeros(self.N_num)               
                n_probs[0] = 1

            if P <= death:
                p_probs = np.zeros(self.P_num)
                p_probs[0] = 1

            if N > death or P > death: # Find normal values where original N and P were not zero
                # Get first f and g
                f1 = r*N - a*N*P
                if logistic:
                    f1 = r*N*(1-N/K) - a*N*P
                g1 = b*P*N - d*P

                N_star = N + f1
                P_star = P + g1

                # Get second f and g
                f2 = r * N_star - a * N_star * P_star
                if logistic:
                    f2 = r * N_star*(1-N_star/K) - a * N_star * P_star
                g2 = b * P_star * N_star - d * P_star

                # Update based on steps
                N += (1 / 2.0) * (f1 + f2)
                P += (1 / 2.0) * (g1 + g2)

                if cull == 'prey':
                    N = N - N_cull

                if cull == 'predator':
                    P = P - P_cull

                if cull == 'both':
                    N = N - N_cull
                    P = P - P_cull
                
                # Ensure base values are positive for log-space math
                N = max(N, 1e-9)
                P = max(P, 1e-9)

                v_linear_interp = np.vectorize(_linear_interp)

                n_probs = v_linear_interp(N_range, N, sigma, self.box_size, self.N_max)
                p_probs = v_linear_interp(P_range, P, sigma, self.box_size, self.P_max)

            joint = np.outer(n_probs, p_probs)
            joint_probs = np.array([
                joint[int(n / self.box_size), int(p / self.box_size)]
                for n, p in self.S_mat
                ])
            mat[i, :] = joint_probs
        self.T_mats[matrix_name] = mat
        if cull == 'prey':
            self.cull_values[cull].append(N_cull)
        elif cull == 'predator':
            self.cull_values[cull].append(P_cull)

    def model_pop(
        self,
        N_0 = None,
        P_0 = None,
        time=200,
        stop_on_pred_ext: bool = True,
        logistic: bool = False,
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
        self.N_list = []
        self.P_list = []

        death, a, r, b, d, sigma, K = self.death, self.a, self.r, self.b, self.d, self.sigma, self.K

        if N_0 == None and P_0 == None:
            N_0 = random.uniform(int(np.round(death))+1, self.N_max)
            P_0 = random.uniform(int(np.round(death))+1, self.P_max)

        N = [N_0]
        P = [P_0]

        for t in range(1, time):
            # stochasticity
            Z_N = stats.lognorm.rvs(loc=0, s=sigma)
            Z_P = stats.lognorm.rvs(loc=0, s=sigma)

            N_t, P_t = N[-1], P[-1]

            # Get first f and g
            f1 = r*N_t - a*N_t*P_t
            if logistic:
                f1 = r*N_t*(1-N_t/K) - a*N_t*P_t
            g1 = b*P_t*N_t - d*P_t

            N_star = N_t + f1
            P_star = P_t + g1

            # Get second f and g
            f2 = r * N_star - a * N_star * P_star
            if logistic:
                f2 = r * N_star*(1-N_star/K) - a * N_star * P_star
            g2 = b * P_star * N_star - d * P_star

            # Update based on steps
            N.append(Z_N*(N_t + (1 / 2.0) * (f1 + f2)))
            P.append(Z_P*(P_t + (1 / 2.0) * (g1 + g2)))

            # extinction check
            if N[t-1] <= death or (stop_on_pred_ext and P[t-1] <= death):
                self.N_list.append(N)
                self.P_list.append(P)
                break
        self.N_list.append(N)
        self.P_list.append(P)

    def plot_time_series(self, end:int=None) -> plt.Figure:
        """
        Plot population time series

        :param type: 'managed' or 'unmanaged' population
        :param cull: Whether to plot culling information
        :param i: Index of the population trajectory to plot, defaults to the last one
        """

        if end is None:
            end = len(self.N_list[0])

        fig, ax = plt.subplots()

        ax.plot(self.N_list[-1][0:end], label='prey')
        ax.plot(self.P_list[-1][0:end], label='predator')
        ax.set_xlabel('Time')
        ax.set_ylabel('Population')
        ax.legend()

        return fig

class SDP(PredatorPreySystem):
    def __init__(self,system: PredatorPreySystem):
        # Inherit from PrfedatorPreySystem
        self.N_max = system.N_max
        self.P_max = system.P_max
        self.T_mats = system.T_mats
        self.cull_values = system.cull_values
        self.S_mat = system.S_mat
        self.num_states = system.num_states

        self.death = system.death
        self.a = system.a
        self.K = system.K
        self.r = system.r
        self.b = system.b
        self.d = system.d
        self.sigma = system.sigma
        self.box_size = system.box_size
        
        # New attributes for SDP
        self.extinct = {'managed':[], 'unmanaged':[]}
        self.ext_mats = None
        self.cull_mat = None
        self.N_list = {'managed':[], 'unmanaged':[]}
        self.P_list = {'managed':[], 'unmanaged':[]}
        self.cull_list = []

    def _feasible_index(self, state_idx, cull_type: str ='prey'):
        n = float(self.S_mat[state_idx, 0])
        p = float(self.S_mat[state_idx, 1])

        if cull_type == 'prey':
            return [i for i, h in enumerate(self.cull_values[cull_type]) if h <= n]
        if cull_type == 'predator':
            return [i for i, h in enumerate(self.cull_values[cull_type]) if h <= p]

    def _optimization_algo(
            self,
            backward,
            terminal_val,
            t_max,
            disc_factor,
            t_threshold,
            tol,
            policy_iter,
            value_iter,
            cull: Literal['prey', 'predator'] = None,
        ) -> None:

        # Define some terms:
        N_num = int(self.N_max*(1/self.box_size)+1)
        P_num = int(self.P_max*(1/self.box_size)+1)

        extinct = np.array([
            (self.S_mat[s, 0] <= self.death) or (self.S_mat[s, 1] <= self.death)
                for s in range(self.num_states)
        ])
        def _Q_value(s, h, V_old):                     
            T = list(self.T_mats.items())[h][1][s] # Get our transition probability

            return self.R_mat[s][h] + disc_factor * (T @ V_old)

        if t_max is None and backward == True:
            raise ValueError("t_max must be specified for backward iteration.")
        
        # Backward iteration to find optimal policy
        if backward == True:
            self.backward = True
            self.A_mat = np.zeros((self.num_states, t_max))
            V = terminal_val # terminal value
            for t in range(t_max):
                V_old = V.copy()
                # V_old[extinct_states] = 0 # Extinct states have large penalty
                for s in range(self.num_states):
                    Q = []
                    feasible_index = self._feasible_index(s, cull_type=cull)
                    for h in feasible_index:
                        Q.append(_Q_value(s, h, V_old))
                    V[s] = max(Q) # Get values
                    self.A_mat[s,t_max - t -1] = self.cull_values[cull][feasible_index[np.argmax(Q)]] # Get indices (policies)
                if t > t_threshold:
                    if value_iter:
                        distance = np.max(np.abs(V - V_old))
                        if distance <= tol:
                            self.stop = t
                            for s in range(self.num_states):
                                self.A_mat[s,0] = self.A_mat[s,t_max - t -1]
                            break
                    if policy_iter:
                        if np.array_equal(self.A_mat[:, t_max-t], self.A_mat[:, t_max-t-1]):
                            self.stop = t
                            for s in range(self.num_states):
                                self.A_mat[s,0] = self.A_mat[s,t_max - t -1]
                            break
            
            # Save Policy Matrix
            self.P_mat = np.zeros((N_num, P_num))

            for s in range(self.num_states):
                n = int(self.S_mat[s, 0]/self.box_size)
                p = int(self.S_mat[s, 1]/self.box_size)
                self.P_mat[n, p] = self.A_mat[s, 0]

        if backward == False:
            self.backward = False
            t = 0
            self.A_mat = np.zeros((self.num_states, 1))
            V = terminal_val
            running = True
            self.P_mat = np.zeros((N_num, P_num))
            while running:
                V_old = V.copy()
                # V_old[extinct_states] = 0 # Extinct states have large penalty
                A_mat_copy = np.zeros((self.num_states, 1))
                for s in range(self.num_states):
                    feasible_index = self._feasible_index(s, cull_type=cull)
                    Q = []
                    for h in feasible_index: # CHANGE NUM ACTIONS
                        Q.append(_Q_value(s, h, V_old))
                    V[s] = max(Q) # Get values
                    A_mat_copy[s, 0] = self.cull_values[cull][feasible_index[np.argmax(Q)]] # Get indices (policies)
                self.A_mat = np.hstack((self.A_mat, A_mat_copy))
                # Save Policy Matrix
                for s in range(self.num_states):
                    n = int(round(self.S_mat[s, 0]/self.box_size))
                    p = int(round(self.S_mat[s, 1]/self.box_size))
                    self.P_mat[n, p] = self.A_mat[s, -1]
                t += 1
                if t > t_threshold:
                    if value_iter:
                        distance = np.max(np.abs(V/np.linalg.norm(V) - V_old/np.linalg.norm(V_old)))
                        print(f"Distance: {distance}")
                        if distance < tol:
                            self.stop = t
                            running = False
                            break
                    if policy_iter:
                        if np.array_equal(self.A_mat[:, t], self.A_mat[:, t-1]):
                            self.stop = t
                            running = False
                            break

    def sdp_biomass(
            self,
            t_max: int=None,
            tol:float = 0,
            t_threshold:float = 10,
            backward=True,
            value_iter = True,
            policy_iter = False,
            disc_factor:float = 1,
            cull = 'prey'
        ):
        """
        Perform stochastic dynamic programming to find optimal culling strategy. This was taken from (3) Munch et al.
        Very interesting stuff considering everything. Only change was that I made the value multiplied instead. Better Results.
        I have to think about hte intution as to why this works.

        JODY: DISREGARD, CURRENTLY BROKEN

        :param t_max: Maximum time horizon
        """
        self.backward = backward
        self.stop = None
        self.cull_total = 0

        self.R_mat = []
        for s in range(self.num_states):
            # Get the feasible actions
            feasible_indeces = self._feasible_index(s, cull_type=cull)

            # Let the R_mat entry be the length of the feasible actions
            self.R_mat.append([0] * len(self.cull_values[cull]))
            N_i =  float(self.S_mat[s,0]) # Initialize population size
            P_i =  float(self.S_mat[s,1])
            for h in feasible_indeces:
                if N_i - self.cull_values[cull][h] <= self.death or P_i <= self.death:
                    self.R_mat[s][h] = 0
                    continue
                else:
                    self.R_mat[s][h] = N_i - self.cull_values[cull][h]

        self._optimization_algo(
            backward,
            self.S_mat.sum(axis=1),
            t_max,
            disc_factor,
            t_threshold,
            tol,
            policy_iter,
            value_iter,
            cull,
        )

    def sdp_pred_biomass(
            self,
            t_max: int=None,
            tol:float = 0,
            t_threshold:float = 10,
            backward=True,
            value_iter = True,
            policy_iter = False,
            disc_factor:float = 1,
            cull = 'prey'
        ):
        """
        Perform stochastic dynamic programming to find optimal culling strategy. This was taken from (3) Munch et al.
        Very interesting stuff considering everything. Only change was that I made the value multiplied instead. Better Results.
        I have to think about hte intution as to why this works.

        JODY: DISREGARD, CURRENTLY BROKEN

        :param t_max: Maximum time horizon
        """
        self.backward = backward
        self.stop = None
        self.cull_total = 0

        self.R_mat = []
        for s in range(self.num_states):
            # Get the feasible actions
            feasible_indeces = self._feasible_index(s, cull_type=cull)

            # Let the R_mat entry be the length of the feasible actions
            self.R_mat.append([0] * len(self.cull_values[cull]))
            N_i =  float(self.S_mat[s,0]) # Initialize population size
            P_i =  float(self.S_mat[s,1])
            for h in feasible_indeces:
                if N_i <= self.death or P_i - self.cull_values[cull][h] <= self.death:
                    self.R_mat[s][h] = 0
                    continue
                else:
                    self.R_mat[s][h] = P_i - self.cull_values[cull][h]

        self._optimization_algo(
            backward,
            self.S_mat.sum(axis=1),
            t_max,
            disc_factor,
            t_threshold,
            tol,
            policy_iter,
            value_iter,
            cull,
        )

    def sdp_sustainable_yield(
            self,
            t_max:int=None,
            disc_factor:float = 1,
            policy_iter = True,
            value_iter = False,
            tol=0,
            t_threshold=10,
            backward=False,
            cull: Literal['prey', 'predator'] = None,
        )-> None:
        '''
        Rewards harvesting the most prey over a large time horizon, with a discount factor.
        '''

        self.stop = None
        self.cull_total = 0

        self.R_mat = []
        for s in range(self.num_states): # Create feasible actions for each state
            feasible_index = self._feasible_index(s, cull_type = cull)
            self.R_mat.append([0] * len(self.cull_values[cull]))
            for h in feasible_index:
                if self.S_mat[s,0] - self.cull_values[cull][h]<= self.death or self.S_mat[s,1] <= self.death:
                    self.R_mat[s][h] = 0
                else:
                    self.R_mat[s][h] = self.cull_values[cull][h] 
        
        self._optimization_algo(
            backward,
            np.zeros((self.num_states,)),
            t_max,
            disc_factor,
            t_threshold,
            tol,
            policy_iter,
            value_iter,
            cull,
        )    

    def sdp_non_extinction(
                self,
                t_max:int=None,
                disc_factor:float = 1,
                policy_iter = True,
                value_iter = False,
                tol=0,
                t_threshold=10,
                backward=False,
                cull: Literal['prey', 'predator'] = None,
            )-> None:
            '''
            Rewards harvesting enough prey to sustain human population
            '''
    
            self.stop = None
            self.cull_total = 0
    
            self.R_mat = []
            for s in range(self.num_states): # Create feasible actions for each state
                feasible_index = self._feasible_index(s, cull_type = cull)
    
                self.R_mat.append([0] * len(self.cull_values[cull]))
                
                for h in feasible_index:
                    if self.S_mat[s,0] - self.cull_values[cull][h] <= self.death or self.S_mat[s,1] <= self.death:
                        self.R_mat[s][h] = -1e9
                    else:
                        self.R_mat[s][h] = 0
            
            self._optimization_algo(
                backward,
                self.S_mat.sum(axis=1),
                t_max,
                disc_factor,
                t_threshold,
                tol,
                policy_iter,
                value_iter,
                cull,
            )    

    def model_pop(
        self,
        N_0 = None,
        P_0 = None,
        time=200,
        stop_on_pred_ext: bool = True,
        cull: Literal['prey', 'predator'] = None,
        logistic: bool = False,
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
        # Set maximums
        N_num = int(self.N_max*(1/self.box_size)+1)
        P_num = int(self.P_max*(1/self.box_size)+1)
        # Set up how to round
        def _index_picker(nbr, max_size):
            idx = int(np.rint(nbr / self.box_size))
            return max(0, min(idx, max_size - 1))
        
        if self.ext_mats == None:
            self.ext_mats = {'managed': [np.zeros((N_num,P_num))
                                        , np.zeros((N_num,P_num))], 
                            'unmanaged':[np.zeros((N_num,P_num))
                                        , np.zeros((N_num,P_num))]}
        if self.cull_mat is None:
            self.cull_mat = np.zeros((N_num,P_num))
        if N_0 == None and P_0 == None:
            N_0 = random.uniform(d+1, self.N_max)
            P_0 = random.uniform(d+1, self.P_max)

        death, a, r, b, d, sigma, K = self.death, self.a, self.r, self.b, self.d, self.sigma, self.K

        if N_0 == None and P_0 == None:
            N_0 = random.uniform(int(np.round(death))+1, self.N_max)
            P_0 = random.uniform(int(np.round(death))+1, self.P_max)

        N = [N_0]
        P = [P_0]
        cull_amts = [0]
        for t in range(1, time):
            # stochasticity
            Z_N = random.lognormvariate(0, sigma)
            Z_P = random.lognormvariate(0, sigma)

            N_t, P_t = N[-1], P[-1]

            # Get first f and g
            f1 = r*N_t - a*N_t*P_t
            if logistic == True:
                f1 = r*N_t*(1-N_t/K) - a*N_t*P_t
            g1 = b*P_t*N_t - d*P_t

            N_star = N_t + f1
            P_star = P_t + g1

            # Get second f and g
            f2 = r * N_star - a * N_star * P_star
            if logistic == True:
                f2 = r * N_star * (1- N_star/K) - a * N_star * P_star
            g2 = b * P_star * N_star - d * P_star

            # Get N and P next
            N_next = N_t + (1 / 2.0) * (f1 + f2)
            P_next = P_t + (1 / 2.0) * (g1 + g2)

            # Check if we need to cull
            if cull:
                if N[t-1] > self.N_max and P[t-1] > self.P_max:
                    cull_amt_next = self.P_mat[N_num - 1, P_num - 1]
                    if cull == 'prey':
                        N_next += -cull_amt_next
                    elif cull == 'predator':
                        P_next += -cull_amt_next
                    self.cull_total += cull_amt_next
                    self.cull_mat[_index_picker(N_0, N_num), _index_picker(P_0,P_num)] += cull_amt_next
                if N[t-1] > self.N_max:
                    cull_amt_next = self.P_mat[N_num - 1, _index_picker(P[t-1],P_num)]
                    if cull == 'prey':
                        N_next += -cull_amt_next
                    elif cull == 'predator':
                        P_next += -cull_amt_next
                    self.cull_total += cull_amt_next
                    self.cull_mat[_index_picker(N_0, N_num), _index_picker(P_0,P_num)] += cull_amt_next
                if P[t-1] > self.P_max:
                    cull_amt_next = self.P_mat[_index_picker(N[t-1],N_num), P_num - 1]
                    if cull == 'prey':
                        N_next += -cull_amt_next
                    elif cull == 'predator':
                        P_next += -cull_amt_next
                    self.cull_total += cull_amt_next
                    self.cull_mat[_index_picker(N_0, N_num), _index_picker(P_0,P_num)] += cull_amt_next
                else:
                    cull_amt_next = self.P_mat[_index_picker(N[t-1], N_num), _index_picker(P[t-1], P_num)]
                    if cull == 'prey':
                        N_next += -cull_amt_next
                    elif cull == 'predator':
                        P_next += -cull_amt_next
                    self.cull_total += cull_amt_next
                    self.cull_mat[_index_picker(N_0, N_num), _index_picker(P_0,P_num)] += cull_amt_next
                cull_amts.append(cull_amt_next)
                
                
            N.append(Z_N*N_next)
            P.append(Z_P*P_next)

            # extinction check
            if N[t-1] <= death or (stop_on_pred_ext and P[t-1] <= death):
                if cull:
                    self.extinct['managed'].append(len(N))
                    self.N_list['managed'].append(N)
                    self.P_list['managed'].append(P)
                    self.cull_list.append(cull_amts)
                    self.ext_mats['managed'][0][_index_picker(N_0, N_num),_index_picker(P_0,P_num)] += 1
                    self.ext_mats['managed'][1][_index_picker(N_0, N_num),_index_picker(P_0,P_num)] += t
                    break
                else:
                    self.extinct['unmanaged'].append(len(N))
                    self.N_list['unmanaged'].append(N)
                    self.P_list['unmanaged'].append(P)
                    self.ext_mats['unmanaged'][0][_index_picker(N_0, N_num),_index_picker(P_0,P_num)] += 1
                    self.ext_mats['unmanaged'][1][_index_picker(N_0, N_num),_index_picker(P_0,P_num)] += t
                    break
        if cull:
            self.N_list['managed'].append(N)
            self.P_list['managed'].append(P)
            self.cull_list.append(cull_amts)
        else:
            self.N_list['unmanaged'].append(N)
            self.P_list['unmanaged'].append(P)
    
    def reset_lists(self):
        """
        Reset the lists used to store population trajectories and extinction times
        """
        self.extinct = {'managed':[], 'unmanaged':[]}
        self.ext_mats = None
        self.cull_mat = None
        self.N_list = {'managed':[], 'unmanaged':[]}
        self.P_list = {'managed':[], 'unmanaged':[]}
        self.cull_total = 0

    def plot_sdp(self, idx=None)-> plt.Figure:
        '''
        Plot an optimal culling strategy

        Actions:
        0 - No cull
        1 - Cull predator
        2 - Cull prey
        '''

        N_num = int((self.N_max/self.box_size)+1)
        P_num = int((self.P_max/self.box_size)+1)
        N_vals = np.linspace(0, self.N_max, N_num)
        P_vals = np.linspace(0, self.P_max, P_num)

        if idx is not None:            
            temp_P_mat = np.zeros((N_num, P_num))

            for s in range(self.num_states):
                n = int(self.S_mat[s, 0]/self.box_size)
                p = int(self.S_mat[s, 1]/self.box_size)
                temp_P_mat[n, p] = self.A_mat[s, idx]

            fig, ax = plt.subplots(figsize=(8, 6))
            c = ax.pcolor(N_vals, P_vals, temp_P_mat.T, shading='auto')
            fig.colorbar(c, ax = ax, label="Action")
            ax.set_xlabel("N")
            ax.set_ylabel("P")
            ax.set_title("Optimal Action Map")
            return fig
    
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            c = ax.pcolor(N_vals, P_vals, self.P_mat.T, shading='auto')
            fig.colorbar(c, ax = ax, label="Action")
            ax.set_xlabel("N")
            ax.set_ylabel("P")
            ax.set_title("Optimal Action Map")
            return fig
    
    def plot_sdp_timelapse(self) -> FuncAnimation:
        """
        Plot a timelapse of the optimal culling strategy evolving over iterations.
        """
        # Get values for SDP
        N_num = int((self.N_max/self.box_size)+1)
        P_num = int((self.P_max/self.box_size)+1)
        N_vals = np.linspace(0, self.N_max, N_num)
        P_vals = np.linspace(0, self.P_max, P_num)

        stop = self.A_mat.shape[1] if self.stop is None else self.stop

        fig, ax = plt.subplots(figsize=(8, 6))

        # Initial matrix
        temp_P_mat = np.zeros((N_num, P_num))

        # Fixed color scale across all frames
        vmin = np.min(self.A_mat)
        vmax = np.max(self.A_mat)

        # Create the mesh once
        mesh = ax.pcolor(
            N_vals,
            P_vals,
            temp_P_mat.T,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        fig.colorbar(mesh, ax=ax)

        ax.set_xlabel("N")
        ax.set_ylabel("P")

        if self.backward == False:
            def update(frame):
                temp_P_mat.fill(0)

                for i in range(self.num_states):
                    n = int(self.S_mat[i, 0]/self.box_size)
                    p = int(self.S_mat[i, 1]/self.box_size)

                    # Guard against invalid indices
                    if 1 <= n <= N_num and 1 <= p <= P_num:
                        temp_P_mat[n, p] = self.A_mat[i, frame]

                # QuadMesh expects a flattened array
                mesh.set_array(temp_P_mat.T.ravel())

                ax.set_title(f"Optimal Action Map at Iteration {frame}")

                return (mesh,)
        else:
            def update(frame):
                temp_P_mat.fill(0)

                for i in range(self.num_states):
                    n = int(self.S_mat[i, 0]/self.box_size)
                    p = int(self.S_mat[i, 1]/self.box_size)

                    # Guard against invalid indices
                    if 1 <= n <= N_num and 1 <= p <= P_num:
                        temp_P_mat[n, p] = self.A_mat[i, stop - frame - 1]

                # QuadMesh expects a flattened array
                mesh.set_array(temp_P_mat.T.ravel())

                ax.set_title(f"Optimal Action Map at Iteration {stop - frame - 1}")

                return (mesh,)

        anim = FuncAnimation(
            fig,
            update,
            frames=range(stop),
            repeat=False,
            blit=False,
        )

        return anim
    
    def plot_sdp_with_trajectories(self, i:int):
        fig = self.plot_sdp()
        ax = fig.axes[0]
        ax.plot(self.N_list['managed'][i], self.P_list['managed'][i], lw=1.0, alpha=0.7, color='black')
        ax.scatter(self.N_list['managed'][i][0], self.P_list['managed'][i][0], color='green', s=30, zorder=10)
        ax.scatter(self.N_list['managed'][i][-1], self.P_list['managed'][i][-1], color='red', s=30, zorder=10)
        return fig

    def find_index(self, N_0, P_0):
        '''
        Find the index of population condition
        '''
        for (N,P) in self.S_mat:
            if N == N_0 and P == P_0:
                return int(i)
        raise ValueError("Indeces not found for the given starting populations.")
    
    def find_index_start(self, N_0, P_0):
        '''
        Find the index of the trajectory with the given starting populations
        '''
        for i in range(len(self.N_list['managed'])):
            if self.N_list['managed'][i][0] == N_0 and self.P_list['managed'][i][0] == P_0:
                return i
        raise ValueError("No trajectory found with the given starting populations.")

    def _sdp_dual_harv(
            self,
            h_max:int,
            c_max:int,
            t_max=250,
            disc_factor:float = 1,
            policy_iter = True,
            value_iter = True,
            tol=0,
            t_threshold=10,
            backward=True,
        )-> None:
        '''
        DEPRECATED

        Rewards harvesting the most prey and predators over a large time horizon, with a discount factor.
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
                if N_i <= self.d or P_i <= self.d:
                    self.R_mat[s, a] = 0
                    continue
                self.R_mat[s,a] = min(N_i,h) + min(P_i,c)

        self._optimization_algo(
            backward,
            self.S_mat.prod(axis=1),
            t_max,
            num_actions,
            disc_factor,
            t_threshold,
            tol,
            policy_iter,
            value_iter,
        )

def plot_time_series(sdp:SDP, type:list,cull: bool = False,i=None,end:int=None) -> plt.Figure:
    """
    Plot population time series

    :param type: 'managed' or 'unmanaged' population
    :param cull: Whether to plot culling information
    :param i: Index of the population trajectory to plot, defaults to the last one
    """
    if i is None:
        i = len(sdp.N_list[type]) - 1

    if end is None:
        end = len(sdp.N_list[type][i])

    fig, ax = plt.subplots()

    ax.plot(sdp.N_list[type][i][0:end], label='prey')
    ax.plot(sdp.P_list[type][i][0:end], label='predator')
    if cull and type == 'managed':
        ax.plot(sdp.cull_list[i][0:end], label='cull', color='red')
    ax.set_xlabel('Time')

    ax.legend()

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