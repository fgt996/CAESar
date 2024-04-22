
import numpy as np
import multiprocessing as mp
from models.caviar import CAViaR

class K_CAViaR():
    def __init__(self, theta, spec='AS', n_points=10):
        self.theta = theta
        self.mdl_spec = spec
        self.points = np.linspace(0, theta, n_points+1, endpoint=True)[1:]
    
    def qcaviar_warper(self, y, ti, theta_j, seed, return_train, q0, pipend):
        mdl = CAViaR(theta_j, self.mdl_spec)
        res = mdl.fit_predict(y, ti, seed=seed, return_train=return_train, q0=q0)
        pipend.send(res)
    
    def fit_predict(self, y, ti, seed=2, jobs=4, return_train=False, q0=None):
        # Initialize the list of quantile forecasts at different levels theta_j
        qf_list = list()
        if return_train:
            qi_list = list()

        # Compute CAViaR in the inner theta_j
        for q_start in range(0, len(self.points), jobs):
            # Create and start worker processes
            workers = list() # Initialize the list of workers
            end_point = np.min([q_start+jobs, len(self.points)]) # Define the end point of the iteration
            
            for theta_j in self.points[q_start:end_point]: # Iterate over theta_j
                parent_pipend, child_pipend = mp.Pipe() # Create a pipe to communicate with the worker
                worker = mp.Process(target=self.qcaviar_warper,
                                args=(y, ti, theta_j, seed, return_train, q0, child_pipend)) # Define the worker
                workers.append([worker, parent_pipend]) # Append the worker to the list
                worker.start() # Start the worker

            # Gather results from workers
            for worker, parent_pipend in workers:
                temp_res = parent_pipend.recv() # Get the result from the worker
                worker.join() # Wait for the worker to finish
                qf_list.append(temp_res['qf'])
                if return_train:
                    qi_list.append(temp_res['qi'])
        
        # From list to array
        qf_list = np.array(qf_list)
        if return_train:
            qi_list = np.array(qi_list)
            out_dict = {'qi':qi_list[-1,:], 'ei':np.mean(qi_list, axis=0),
                        'qf':qf_list[-1,:], 'ef':np.mean(qf_list, axis=0)}
        else:
            out_dict = {'qf':qf_list[-1,:], 'ef':np.mean(qf_list, axis=0)}
        return out_dict