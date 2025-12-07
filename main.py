import pandas as pd
from SDP import PredatorPreySDP
from IPython.display import HTML
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle

def run(inits:list = None, args:list = None) -> None:
    sim = PredatorPreySDP(*inits)
    sim.run(*args, matrix_name="T Matrix", sims=300, time = 3)
    sim.run(*args, matrix_name="T N Matrix 1", time = 3, sims=100, cull = 'prey', N_cull= 1)
    sim.run(*args ,matrix_name="T N Matrix 2",  sims=100, time = 3,cull = 'prey', N_cull= 2)
    sim.run(*args, matrix_name="T N Matrix 3",  sims=100, time = 3,cull = 'prey', N_cull= 3)
    sim.run(*args, matrix_name="T N Matrix 4",  sims=100, time = 3,cull = 'prey', N_cull= 4)
    sim.run(*args, matrix_name="T N Matrix 5",  sims=100, time = 3,cull = 'prey', N_cull= 5)
    sim.sdp()

    # Customize this eventually, for now, unable
    for i in range(sim.num_states):
        for _ in range(100):
            sim.model_pop(N_0 = int(sim.S_mat[i,0]), P_0=int(sim.S_mat[i,1]))

    for i in range(sim.num_states):
        for _ in range(100):
            sim.model_pop(cull = False, N_0 = int(sim.S_mat[i,0]), P_0=int(sim.S_mat[i,1]))

    return sim

@st.cache_resource
def load_sim() -> PredatorPreySDP:
    with open("SDP_simulation.pkl", "rb") as f:
        sim = pickle.load(f)
    return sim

if __name__ == "__main__":
    if "ran_default_sim" not in st.session_state:
        st.session_state["ran_default_sim"] = False
    if "ran_new_sim" not in st.session_state:
        st.session_state["ran_new_sim"] = False
    if "ran_sim" not in st.session_state:
        st.session_state["ran_sim"] = False
    if "time_series_index" not in st.session_state:
        st.session_state['time_series_index'] = 92941

    with st.sidebar:
        st.title("Control Bar")
        if st.button("Load Default Simulation"):
            st.session_state['ran_sim'] = True
            st.session_state['ran_default_sim'] = True
        st.divider()
        if st.button("Run New Simulation"):
            st.session_state['ran_sim'] = True
            st.session_state['ran_new_sim'] = True
        N_max = st.slider('Max N', 0, 100, 40)
        P_max = st.slider('Max P', 0, 100, 30)

        K = st.number_input('K', 0, 1000, 100)
        d = st.slider('d', 0, 10, 2)
        a = st.slider('a', 0.0, 1.0, 0.01)
        r = st.slider('r', 0.0, 2.5, 1.3)
        b = st.slider('b', 0.0, 1.0, 0.08)
        mu = st.slider('mu', 0.0, 1.0, 0.1)
        sigma = st.slider('sigma', 0.0, 1.0, 0.05)

        on = st.toggle("Display Guide")
        if on:
            st.info(
                """
                d - Minimum value of the population. If below d, pop will extinct

                a - The success rate of hunts by predator population

                K - Carrying capacity of prey

                r - The growth rate of prey

                b - The birth rate of predators (reliant on prey pop)

                mu - The death reate of predators

                sigma - The environmental stochasticity factor
                """
        )
        
        inits = [N_max,P_max]
        args = [d,a,K,r,b,mu,sigma]

    st.title("Predator Prey SDP")
    if st.toggle("Show Class Documentation", key="show_doc"):
        st.help(PredatorPreySDP)
    if st.session_state["ran_sim"] == True:
        if st.session_state['ran_default_sim'] == True:
            sim = load_sim()
        if st.session_state['ran_new_sim'] == True:
            sim = run(inits, args)

        opt_plot = sim.plot_sdp()
        st.pyplot(opt_plot)

        # Create two columns
        st.header('Time Series Population Plots')
        col1, col2 = st.columns(2)
        with col1:
            N_0 = st.number_input(
                    'Inital N_0', 
                    min_value=0,
                    max_value = sim.N_max, 
                    value=20, #default
                    step=1)
        with col2:
            P_0 = st.number_input(
                    'Inital P_0', 
                    min_value=0,
                    max_value = sim.P_max, 
                    value=10, #default
                    step=1)
        
        # Create indeces for given N_0 and P_)
        man_initial_indices = []
        for idx, (n, p) in enumerate(zip(sim.N_list['managed'], sim.P_list['managed'])):
            if n[0] == N_0 and p[0] == P_0:
                man_initial_indices.append(idx)
        unman_initial_indices = []
        for idx, (n, p) in enumerate(zip(sim.N_list['unmanaged'], sim.P_list['unmanaged'])):
            if n[0] == N_0 and p[0] == P_0:
                unman_initial_indices.append(idx)


        col1, col2 = st.columns(2)
        with col1:
            st.header(f'Unmanaged Time Series')
            unman_number = st.selectbox("Index", unman_initial_indices)
            unman_time_plot = sim.plot(type='unmanaged', i=unman_number)
            st.pyplot(unman_time_plot)

        with col2:
            st.header('Managed Time Series')
            man_number = st.selectbox("Index", man_initial_indices)
            man_time_plot = sim.plot(type='managed', i=man_number)
            st.pyplot(man_time_plot)

        st.header('Percent Extinction Matrices')
        col1, col2, col3 = st.columns(3)

        # Unmanaged extinction matrix
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            c1 = ax1.pcolor(
                np.arange(sim.N_max+1),
                np.arange(sim.P_max+1),
                sim.ext_mats['unmanaged'][0].T / 100,
                shading='auto'
            )
            fig1.colorbar(c1, ax=ax1, label="Extinction %")
            ax1.set_xlabel("N₀")
            ax1.set_ylabel("P₀")
            ax1.set_title("Unmanaged Extinction")
            st.pyplot(fig1)

        # Managed extinction matrix
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            c2 = ax2.pcolor(
                np.arange(sim.N_max+1),
                np.arange(sim.P_max+1),
                sim.ext_mats['managed'][0].T / 100,
                shading='auto'
            )
            fig2.colorbar(c2, ax=ax2, label="Extinction %")
            ax2.set_xlabel("N₀")
            ax2.set_ylabel("P₀")
            ax2.set_title("Managed Extinction")
            st.pyplot(fig2)

        # Difference matrix
        with col3:
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            c3 = ax3.pcolor(
                np.arange(sim.N_max+1),
                np.arange(sim.P_max+1),
                (sim.ext_mats['unmanaged'][0] - sim.ext_mats['managed'][0]).T / 100,
                shading='auto'
            )
            fig3.colorbar(c3, ax=ax3, label="Extinction Difference %")
            ax3.set_xlabel("N₀")
            ax3.set_ylabel("P₀")
            ax3.set_title("Difference Matrix")
            st.pyplot(fig3)

        st.header('Average Extinction Time Matrices')
        col1, col2 = st.columns(2)

        # Unmanaged extinction matrix
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            c1 = ax1.pcolor(
                np.arange(sim.N_max+1),
                np.arange(sim.P_max+1),
                sim.ext_mats['unmanaged'][1].T / sim.ext_mats['unmanaged'][0].T,
                shading='auto'
            )
            fig1.colorbar(c1, ax=ax1, label="Average Extinction Time")
            ax1.set_xlabel("N₀")
            ax1.set_ylabel("P₀")
            ax1.set_title("Unmanaged Extinction")
            st.pyplot(fig1)

        # Managed extinction matrix
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            c2 = ax2.pcolor(
                np.arange(sim.N_max+1),
                np.arange(sim.P_max+1),
                sim.ext_mats['managed'][1].T / sim.ext_mats['managed'][0].T,
                shading='auto'
            )
            fig2.colorbar(c2, ax=ax2, label="Average Extinction Time")
            ax2.set_xlabel("N₀")
            ax2.set_ylabel("P₀")
            ax2.set_title("Managed Extinction")
            st.pyplot(fig2)

        # Difference matrix - Not ready yet, fix NaN to by infinity
        # with col3:
        #     fig3, ax3 = plt.subplots(figsize=(6, 5))
        #     c3 = ax3.pcolor(
        #         np.arange(sim.N_max+1)
        #         , np.arange(sim.P_max+1)
        #         , (sim.ext_mats['unmanaged'][1] - sim.ext_mats['managed'][1]).T/
        #             (sim.ext_mats['unmanaged'][0] - sim.ext_mats['managed'][0]).T
        #         , shading='auto'
        #     )
        #     fig3.colorbar(c3, ax=ax3, label="Average Extinction Time Difference")
        #     ax3.set_xlabel("N₀")
        #     ax3.set_ylabel("P₀")
        #     ax3.set_title("Difference Matrix")
        #     st.pyplot(fig3)