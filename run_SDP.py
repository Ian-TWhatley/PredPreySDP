from SDP import PredatorPreySDP
from tqdm import tqdm
import pickle

sim = PredatorPreySDP(N_max = 40, P_max = 30)
sim.run(sims=100, matrix_name="T Matrix", time = 3)
sim.run(sims=100, matrix_name="T N Matrix 1", time = 3,cull = 'prey', N_cull= 1)
sim.run(sims=100, matrix_name="T N Matrix 2", time = 3,cull = 'prey', N_cull= 2)
sim.run(sims=100, matrix_name="T N Matrix 3", time = 3,cull = 'prey', N_cull= 3)
sim.run(sims=100, matrix_name="T N Matrix 4", time = 3,cull = 'prey', N_cull= 4)
sim.run(sims=100, matrix_name="T N Matrix 5", time = 3,cull = 'prey', N_cull= 5)
sim.run(sims=100, matrix_name="T N Matrix 6", time = 3,cull = 'prey', N_cull= 6)
sim.run(sims=100, matrix_name="T N Matrix 7", time = 3,cull = 'prey', N_cull= 7)
sim.run(sims=100, matrix_name="T N Matrix 8", time = 3,cull = 'prey', N_cull= 8)
sim.run(sims=100, matrix_name="T N Matrix 9", time = 3,cull = 'prey', N_cull= 9)
sim.run(sims=100, matrix_name="T N Matrix 10", time = 3,cull = 'prey', N_cull= 10)

file_path = 'SDP_simulation.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(sim, file)