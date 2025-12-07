import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class PredatorPreySDP:
    def __init__(self, N_max = 40, P_max=30):
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
        ):
        """
        d - Minimum value of the population. If below d, pop will extinct
        a - The success rate of hunts by predator population
        K - Carrying capacity of prey
        r - The growth rate of prey
        b - The birth rate of predators (reliant on prey pop)
        mu - The death reate of predators
        sigma - SD of stochastic noise
        time - the amount of time to run the simulation

        cull = ['none', 'prey','predator']
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

    def mat_normalize(self,mat):
        row_sums = mat.sum(axis=1, keepdims=True)
        return np.divide(mat, row_sums, where=row_sums!=0)

    def sdp(self, t_max:int = 250):
        self.t_max = t_max
        V = self.S_mat**2

        V[:,0] = V[:,0]/max(V[:,0])
        V[:,1] = V[:,1]/max(V[:,1])
        V = V.sum(axis=1) # terminal value 

        self.A_mat = np.zeros((self.num_states, self.t_max), dtype=int)
        # Backward iteration
        for t in range(self.t_max):
            for i in range(len(list(self.T_mats.items()))):
                if i == 0:
                    tup = (list(self.T_mats.items())[i][1] @ V,)
                else:
                    tup = tup + (list(self.T_mats.items())[i][1] @ V,)
            cull = np.column_stack(tup)

            V = np.max(cull, axis=1)
            self.A_mat[:, self.t_max - t -1] = np.argmax(cull,axis=1)

    def reset_lists(self):
        self.extinct = {'managed':[], 'unmanaged':[]}
        self.ext_mats = None
        self.N_list = {'managed':[], 'unmanaged':[]}
        self.P_list = {'managed':[], 'unmanaged':[]}

    def extinction_matrices(
            self,
            reset:bool= True,
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
            sims =1,
        ):
        
        if self.ext_mats == None or reset:
            self.ext_mats = {'managed': np.zeros((self.N_max+1,self.P_max+1)), 'unmanaged':np.zeros((self.N_max+1,self.P_max+1))}

        if cull:
            P_mat = np.zeros((self.N_max, self.P_max))

            for i in range(self.num_states):
                n = int(self.S_mat[i, 0])
                p = int(self.S_mat[i, 1])
                P_mat[n-1, p-1] = self.A_mat[i, 0]

        for i in range(self.num_states):
            N_0 = int(self.S_mat[i,0])
            P_0 = int(self.S_mat[i,1])
            N = [N_0]
            P = [P_0]
            for _ in sims:
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
                            N_next += -P_mat[self.N_max - 1, self.P_max - 1]
                        if N[t-1] > self.N_max:
                            N_next += -P_mat[self.N_max - 1, round(P[t-1])-1]
                        if P[t-1] > self.P_max:
                            N_next += -P_mat[round(N[t-1])-1, self.P_max-1]
                        else:
                            N_next += -P_mat[round(N[t-1])-1, round(P[t-1])-1]

                    N.append(round(Z_N * N_next))
                    P.append(round(Z_P * P_next))

                    # extinction check
                    if N[t] < d or P[t] < d:
                        if cull:
                            self.extinct['managed'].append(len(N))
                            self.N_list['managed'].append(N)
                            self.P_list['managed'].append(P)
                            self.ext_mats['managed'][N_0,P_0] += 1
                            break
                        else:
                            self.extinct['unmanaged'].append(len(N))
                            self.N_list['unmanaged'].append(N)
                            self.P_list['unmanaged'].append(P)
                            self.ext_mats['unmanaged'][N_0,P_0] += 1
                            break
                if cull:
                    self.N_list['managed'].append(N)
                    self.P_list['managed'].append(P)
                else:
                    self.N_list['unmanaged'].append(N)
                    self.P_list['unmanaged'].append(P)

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
        ):
        
        if self.ext_mats == None:
            self.ext_mats = {'managed': np.zeros((self.N_max+1,self.P_max+1)), 'unmanaged':np.zeros((self.N_max+1,self.P_max+1))}

        if cull:
            P_mat = np.zeros((self.N_max, self.P_max))

            for i in range(self.num_states):
                n = int(self.S_mat[i, 0])
                p = int(self.S_mat[i, 1])
                P_mat[n-1, p-1] = self.A_mat[i, 0]

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
                    N_next += -P_mat[self.N_max - 1, self.P_max - 1]
                if N[t-1] > self.N_max:
                    N_next += -P_mat[self.N_max - 1, round(P[t-1])-1]
                if P[t-1] > self.P_max:
                    N_next += -P_mat[round(N[t-1])-1, self.P_max-1]
                else:
                     N_next += -P_mat[round(N[t-1])-1, round(P[t-1])-1]

            N.append(round(Z_N * N_next))
            P.append(round(Z_P * P_next))

            # extinction check
            if N[t] < d or P[t] < d:
                if cull:
                    self.extinct['managed'].append(len(N))
                    self.N_list['managed'].append(N)
                    self.P_list['managed'].append(P)
                    self.ext_mats['managed'][N_0,P_0] += 1
                    break
                else:
                    self.extinct['unmanaged'].append(len(N))
                    self.N_list['unmanaged'].append(N)
                    self.P_list['unmanaged'].append(P)
                    self.ext_mats['unmanaged'][N_0,P_0] += 1
                    break
        if cull:
            self.N_list['managed'].append(N)
            self.P_list['managed'].append(P)
        else:
            self.N_list['unmanaged'].append(N)
            self.P_list['unmanaged'].append(P)

    def plot(self, type,i=None):
        if i is None:
            i = len(self.N_list[type]) - 1

        fig, ax = plt.subplots()

        ax.plot(self.N_list[type][i], label='prey')
        ax.plot(self.P_list[type][i], label='predator')

        ax.legend()

        return fig
    
    def plot_sdp(self):
        '''
        Plot an optimal culling strategy
        0 - No cull
        1 - Cull predator
        2 - Cull prey
        '''
        image_mat = np.zeros((self.N_max, self.P_max))

        for i in range(self.num_states):
            n = int(self.S_mat[i, 0])
            p = int(self.S_mat[i, 1])
            image_mat[n-1, p-1] = self.A_mat[i, 0]

        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.pcolor(np.arange(self.N_max+1), np.arange(self.P_max+1), image_mat.T, shading='auto')
        fig.colorbar(c, ax = ax, label="Action")
        ax.set_xlabel("N")
        ax.set_ylabel("P")
        ax.set_title("Optimal Action Map")
        return fig
        