from SDP import PredatorPreySystem
from tqdm import tqdm
import pickle

sim = PredatorPreySystem(N_max = 40, P_max = 30)
sim.run(matrix_name="T Matrix")
for i in tqdm(range(1, 41)):
    sim.run(matrix_name=f"T N Matrix {i}", cull = 'prey', N_cull= i)

file_path = 'SDP_default_simulation.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(sim, file)