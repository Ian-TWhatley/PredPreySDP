import numpy as np
import random
import matplotlib.pyplot as plt

class PredatorPreySDP:
    """
    Discrete predator-prey ODE simulation with stochastic noise and a stochastic dynamic programming component

    Attributes:
        extinct (dict): Dictionary to store extinction times for managed and unmanaged populations
        ext_mats (dict): Matrices to store extinction counts and times
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
        self.extinct = {'managed':[], 'unmanaged':[]}
        self.ext_mats = None
        self.N_list = {'managed':[], 'unmanaged':[]}
        self.P_list = {'managed':[], 'unmanaged':[]}
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
            time=200,
            sims = 1,
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
        :param cull: Type of culling ('prey', 'predator', 'none')
        :param N_cull: Number of prey to cull if culling prey
        :param P_cull: Number of predators to cull if culling predators
        """
        state_f_ind = 0
        state_i_ind = 0
        mat = np.zeros((self.num_states,self.num_states),dtype=float) # Set matrix, may be anything
        for _ in range(sims):
            for i in range(self.num_states):
                N = [self.S_mat[i,0]]
                P = [self.S_mat[i,1]]

                for t in range(1, time):
                    # Begin starting index for T matrix
                    state_i_ind = np.where(np.all(self.S_mat == np.array([N[-1], P[-1]]), axis=1))[0][0]

                    # Create stochasticity
                    Z_N = random.lognormvariate(0, sigma)
                    Z_P = random.lognormvariate(0, sigma)

                    # prey update
                    N_next = r*N[t-1]*(1 - N[t-1] / K) - a*N[t-1]*P[t-1]

                    # predator update
                    P_next = b*P[t-1]*N[t-1] - mu*P[t-1]

                    if cull == 'prey':
                        N_next = N_next - N_cull

                    if cull == 'predator':
                        P_next = P_next - P_cull

                    N.append(round(min(Z_N * N_next, self.N_max)))
                    P.append(round(min(Z_P * P_next, self.P_max)))

                    # extinction check
                    if N[t] < d or P[t] < d:
                        if N[t] < d: N[t] = 0
                        if P[t] < d: P[t] = 0
                        state_f_ind = np.where(np.all(self.S_mat == np.array([N[-1], P[-1]]), axis=1))[0][0]

                        # Update
                        mat[state_i_ind,state_f_ind] += 1
                        break
                    
                    # maximum check
                    if N[t] > self.N_max or P[t] > self.P_max:
                        if N[t] > self.N_max: N[t] = self.N_max
                        if P[t] > self.P_max: P[t] = self.P_max

                        # Find final index and update T matrix
                        state_f_ind = np.where(np.all(self.S_mat == np.array([N[-1], P[-1]]), axis=1))[0][0]
                        mat[state_i_ind,state_f_ind] += 1
                        break
                    
                    # Find final index and update
                    state_f_ind = np.where(np.all(self.S_mat == np.array([N[-1], P[-1]]), axis=1))[0][0] 
                    mat[state_i_ind,state_f_ind] += 1
                
                self.T_mats[matrix_name] = mat
        for mat in self.T_mats:
            self.T_mats[mat] = self.mat_normalize(self.T_mats[mat])

    def mat_normalize(self,mat) -> np.array:
        """
        Normalize a matrix by row sums
        
        :param mat: Matrix to normalize
        """
        row_sums = mat.sum(axis=1, keepdims=True)
        return np.divide(mat, row_sums, where=row_sums!=0)
    
    def sdp_economic(self, c_base, base_price, t_max:int = 250)-> None:
        """
        Perform stochastic dynamic programming to find optimal culling strategy based on economic incentives
        :param c_base: Base cost of culling per animal
        :param base_price: Base price of harvested animal
        :param t_max: Maximum time horizon
        """
        R_mat = np.zeros((self.num_states,len(list(self.T_mats.items()))))

        for i in range(self.num_states):
            N_i =  float(self.S_mat[i,0])
            v_scarce = 1 + (1 - N_i/self.N_max)  # Scarcity multiplier increases as population decreases
            for j in range(0,len(list(self.T_mats.items()))):
                # R_i,j = Harvest_j * Base_Price * V_scarcity(N_i) - Cost_j
                # We apply the scarcity multiplier (V_scarcity) to the base price
                if N_i == 0:
                    R_mat[i, j] = 0
                    continue
                harv_rev = base_price * v_scarce
                harv_cost = c_base * j
                R_mat[i, j] = harv_rev - harv_cost

        V = self.S_mat
        self.t_max = t_max
        V = V.sum(axis=1) # terminal value 

        self.A_mat = np.zeros((self.num_states, self.t_max), dtype=int)
        
        for t in range(self.t_max):
            for i in range(len(list(self.T_mats.items()))):
                # Calculate Future Value (Expected Value of next state)
                future_val = list(self.T_mats.items())[i][1] @ V
                
                # Add Immediate Economic Reward (R_mat) to Future Value
                # This is the "Economic Incentive" step
                total_val = R_mat[:, i] + future_val

                if i == 0:
                    tup = (total_val,)
                else:
                    tup = tup + (total_val,)
            
            cull = np.column_stack(tup)

            V = np.max(cull, axis=1)
            self.A_mat[:, self.t_max - t -1] = np.argmax(cull,axis=1)
    
        # Save Policy Matrix
        self.P_mat = np.zeros((self.N_max, self.P_max))

        for i in range(self.num_states):
            n = int(self.S_mat[i, 0])
            p = int(self.S_mat[i, 1])
            self.P_mat[n-1, p-1] = self.A_mat[i, 0]

    def sdp_standard(self, t_max:int = 500, tol:float = 1e-8, t_threshold:float = 10)-> None:
        """
        Perform SDP solly based on population sizes of predator and prey

        Perform stochastic dynamic programming to find optimal culling strategy
        :param t_max: Maximum time horizon
        """
        self.stop = None
        V = self.S_mat
        V = V.sum(axis=1) # terminal value

        self.A_mat = np.zeros((self.num_states, t_max), dtype=int)
        # Backward iteration
        for t in range(t_max):
            V_old = V.copy()
            Q = np.zeros((self.num_states, len(list(self.T_mats.items()))))
            for i in range(len(list(self.T_mats.items()))):
                Q[:, i] = list(self.T_mats.items())[i][1] @ V
                
            V = np.max(Q, axis=1)
            self.A_mat[:, t_max - t -1] = np.argmax(Q,axis=1)
            if t > t_threshold:
                if np.linalg.norm(V - V_old) < tol:
                    self.stop = t
                    self.A_mat[:, 0] = np.argmax(Q,axis=1)
                    break
        
        # Save Policy Matrix
        self.P_mat = np.zeros((self.N_max, self.P_max))

        for i in range(self.num_states):
            n = int(self.S_mat[i, 0])
            p = int(self.S_mat[i, 1])
            self.P_mat[n-1, p-1] = self.A_mat[i, 0]

    def sdp_harv(self,t_max=250, disc_factor:float = 0.99,alpha:float = 0.5, tol=1e-6, start_thresh=10)-> None: # This rewards harvesting animals,
        '''
        SDP that rewards harvesting prey and the total population size of predator and prey with
        a weighting factor alpha between the two objectives.

        :param t_max: Maximum time horizon
        :param disc_factor: Discount factor for future rewards
        :param alpha: Weighting factor between harvesting reward and population size reward
        '''
        self.stop = None
        num_actions = len(list(self.T_mats.items()))
        R_mat = np.zeros((self.num_states,len(list(self.T_mats.items()))))
        for i in range(self.num_states):
            N_i =  float(self.S_mat[i,0])
            P_i = float(self.S_mat[i,1])
            for j in range(0,len(list(self.T_mats.items()))):
                if N_i == 0:
                    R_mat[i, j] = 0
                    continue
                R_mat[i, j] = (1-alpha)*min(N_i, j) + alpha * ((N_i-j) + P_i) # Reward is number harvested plus value of remaining population

        V = np.zeros(self.num_states)
        self.A_mat = np.zeros((self.num_states, t_max), dtype=int)
        
        for t in range(t_max):
            V_old = V.copy()
            Q = np.zeros((self.num_states, num_actions))
            for i in range(num_actions):
                # Calculate Future Value (Expected Value of next state)
                future_val = list(self.T_mats.items())[i][1] @ V # This is really V_{t+1}(s')
                
                # Find total value of now and future
                Q[:, i] = R_mat[:, i] + disc_factor*future_val

            V = np.max(Q, axis=1)
            self.A_mat[:, t_max - t -1] = np.argmax(Q,axis=1)
            if t > start_thresh and np.sum(V) > 0 and np.sum(V_old) > 0:
                if np.linalg.norm(V/np.sum(V) - V_old/np.sum(V_old)) < tol:
                    self.stop = t
                    self.A_mat[:, 0] = np.argmax(Q,axis=1)
                    break   
    
        # Save Policy Matrix
        self.P_mat = np.zeros((self.N_max, self.P_max))

        for i in range(self.num_states):
            n = int(self.S_mat[i, 0])
            p = int(self.S_mat[i, 1])
            self.P_mat[n-1, p-1] = self.A_mat[i, 0]
        
    def reset_lists(self) -> None:
        """
        Reset the lists used to store population trajectories and extinction times
        """
        self.extinct = {'managed':[], 'unmanaged':[]}
        self.ext_mats = None
        self.N_list = {'managed':[], 'unmanaged':[]}
        self.P_list = {'managed':[], 'unmanaged':[]}

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
            P_next = round(P_next)

            # check if we need to cull
            if cull:
                if N[t-1] > self.N_max and P[t-1] > self.P_max:
                    N_next += -self.P_mat[self.N_max - 1, self.P_max - 1]
                if N[t-1] > self.N_max:
                    N_next += -self.P_mat[self.N_max - 1, round(P[t-1])-1]
                if P[t-1] > self.P_max:
                    N_next += -self.P_mat[round(N[t-1])-1, self.P_max-1]
                else:
                        N_next += -self.P_mat[round(N[t-1])-1, round(P[t-1])-1]

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

    def plot(self, type:list,tol,i=None) -> plt.Figure:
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
    
    def _plot_extinction_matrices_percents(self) -> plt.Figure:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        # Unmanaged extinction matrix
        c1 = ax1.pcolor(np.arange(self.N_max+1), np.arange(self.P_max+1), self.ext_mats['unmanaged'][0].T/100, shading='auto',)
        fig.colorbar(c1, ax=ax1, label="Extinction %")
        ax1.set_xlabel("N_0")
        ax1.set_ylabel("P_0")
        ax1.set_title("Unmanaged Extinction Matrix")

        # Managed extinction matrix
        c2 = ax2.pcolor(np.arange(self.N_max+1), np.arange(self.P_max+1), self.ext_mats['managed'][0].T/100, shading='auto')
        fig.colorbar(c2, ax=ax2, label="Extinction %")
        ax2.set_xlabel("N_0")
        ax2.set_ylabel("P_0")
        ax2.set_title("Managed Extinction Matrix")

        c3 = ax3.pcolor(np.arange(self.N_max+1), np.arange(self.P_max+1), (self.ext_mats['unmanaged'][0]-self.ext_mats['managed'][0]).T/100, shading='auto')
        fig.colorbar(c3, ax=ax3, label="Extinction Difference %")
        ax3.set_xlabel("N_0")
        ax3.set_ylabel("P_0")
        ax3.set_title("Difference Matrix")

        plt.tight_layout()
        plt.show()

    def _plot_extinction_matrices_times(self) -> plt.Figure:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        # Unmanaged extinction matrix
        c1 = ax1.pcolor(np.arange(self.N_max+1), np.arange(self.P_max+1), self.ext_mats['unmanaged'][1].T/self.ext_mats['unmanaged'][0].T, shading='auto',)
        fig.colorbar(c1, ax=ax1, label="Average Extinction Time")
        ax1.set_xlabel("N_0")
        ax1.set_ylabel("P_0")
        ax1.set_title("Unmanaged Extinction Matrix")

        # Managed extinction matrix
        c2 = ax2.pcolor(np.arange(self.N_max+1), np.arange(self.P_max+1), 
                        self.ext_mats['managed'][1].T/self.ext_mats['managed'][0].T, shading='auto')
        fig.colorbar(c2, ax=ax2, label="Average Extinction Time")
        ax2.set_xlabel("N_0")
        ax2.set_ylabel("P_0")
        ax2.set_title("Managed Extinction Matrix")

        c3 = ax3.pcolor(np.arange(self.N_max+1), np.arange(self.P_max+1), 
                        (self.ext_mats['unmanaged'][1].T-self.ext_mats['managed'][1].T)/
                            (self.ext_mats['unmanaged'][0].T-self.ext_mats['managed'][0].T), shading='auto')
        fig.colorbar(c3, ax=ax3, label="Average Extinction Time Difference")
        ax3.set_xlabel("N_0")
        ax3.set_ylabel("P_0")
        ax3.set_title("Difference Matrix")

        plt.tight_layout()
        plt.show()