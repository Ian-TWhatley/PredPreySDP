from SDP import PredatorPreySystem
from tqdm import tqdm
import pickle

sim = PredatorPreySystem(N_max = 40, P_max = 30)
sim.run(sims=100, matrix_name="T Matrix", time = 3)
for i in tqdm(range(1, 21)):
    sim.run(sims=100, matrix_name=f"T N Matrix {i}", time = 3, cull = 'prey', N_cull= i)

file_path = 'SDP_simulation.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(sim, file)