from SDP import PredatorPreySDP
from tqdm import tqdm

sim = PredatorPreySDP(N_max = 40, P_max = 30)
sim.run(sims=300, matrix_name="T Matrix", time = 3)
sim.run(sims=100, matrix_name="T N Matrix 1", time = 3,cull = 'prey', N_cull= 1)
sim.run(sims=100, matrix_name="T N Matrix 2", time = 3,cull = 'prey', N_cull= 2)
sim.run(sims=100, matrix_name="T N Matrix 3", time = 3,cull = 'prey', N_cull= 3)
sim.run(sims=100, matrix_name="T N Matrix 4", time = 3,cull = 'prey', N_cull= 4)
sim.run(sims=100, matrix_name="T N Matrix 5", time = 3,cull = 'prey', N_cull= 5)

sim.sdp()

for i in tqdm(range(sim.num_states)):
    for _ in range(100):
        sim.model_pop(N_0 = int(sim.S_mat[i,0]), P_0=int(sim.S_mat[i,1]))

for i in tqdm(range(sim.num_states)):
    for _ in range(100):
        sim.model_pop(cull = False, N_0 = int(sim.S_mat[i,0]), P_0=int(sim.S_mat[i,1]))

import pickle

file_path = 'SDP_simulation.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(sim, file)