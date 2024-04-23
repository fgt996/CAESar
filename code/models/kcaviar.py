
import numpy as np
import multiprocessing as mp
from models.caviar import CAViaR

class K_CAViaR():
    '''
    Expected Shortfall estimation via Kratz approach with CAViaR for quantile regression.
    '''
    def __init__(self, theta, spec='AS', n_points=10):
        '''
        Initialization of the K-CAViaR model.
        INPUTS:
            - theta: float
                desired confidence level.
            - spec: str, optional
                specification of the model (SAV, AS, GARCH). Default is AS.
            - n_points: int, optional
                number of points for mean approximation. Default is 10.
        OUTPUTS:
            - None
        '''
        self.theta = theta
        self.mdl_spec = spec
        self.points = np.linspace(0, theta, n_points+1, endpoint=True)[1:]
    
    def qcaviar_wrapper(self, y, ti, theta_j, seed, return_train, q0, pipend):
        '''
        Wrapper function for the CAViaR model.
        INPUTS:
            - y: ndarray
                target time series.
            - ti: int
                train set length.
            - theta_j: float
                quantile level.
            - seed: int or None
                random seed.
            - return_train: bool, optional
                return the train set. Default is False.
            - q0: float
                initial quantile. Default is None.
            - pipend: multiprocessing.connection.Connection
                pipe end for communicating multiprocessing.
        OUTPUTS:
            - None
        '''
        mdl = CAViaR(theta_j, self.mdl_spec)
        res = mdl.fit_predict(y, ti, seed=seed, return_train=return_train, q0=q0)
        pipend.send(res)
    
    def fit_predict(self, y, ti, seed=None, jobs=1, return_train=False, q0=None):
        '''
        Fit and predict the K-CAViaR model.
        INPUTS:
            - y: ndarray
                target time series.
            - ti: int
                train set length.
            - seed: int or None, optional
                random seed. Default is None.
            - jobs: int, optional
                number of parallel jobs. Default is 1.
            - return_train: bool, optional
                return the train set. Default is False.
            - q0: float
                initial quantile. Default is None.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - ei: ndarray
                expected shortfall in the training set (if return_train=True).
            - qf: ndarray
                quantile forecast in the test set.
            - ef: ndarray
                expected shortfall forecast in the test set.
        '''
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
                worker = mp.Process(target=self.qcaviar_wrapper,
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