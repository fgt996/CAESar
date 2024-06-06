
import numpy as np
import multiprocessing as mp
from models.caviar import CAViaR

class CAESar_No_Cross():
    '''
    CAESar model for Expected Shortfall estimation - optimization only with Barrera loss.
    '''
    def __init__(self, theta, spec='AS', lambdas=dict()):
        '''
        Initialization of the CAESar model.
        INPUTS:
            - theta: float
                desired confidence level.
            - spec: str, optional
                specification of the model (SAV, AS, GARCH). Default is AS.
            - lambdas: dict, optional
                lambdas for the soft constraints. Default is {'q':10, 'e':10}.
        OUTPUTS:
            - None.
        '''
        self.theta = theta #Initialize theta
        self.lambdas = {'q':10, 'e':10}
        self.lambdas.update(lambdas) #Initialize lambdas for soft constraints

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 3
            self.joint_loop = self.Joint_SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 4
            self.joint_loop = self.Joint_ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 3
            self.joint_loop = self.Joint_GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')
    
    def joint_loss(self, v, e, y):
        '''
        Compute the loss function (Patton) of the model.
        INPUTS:
            - v: ndarray
                Value at Risk forecast.
            - e: ndarray
                Expected Shortfall forecast.
            - y: ndarray
                target time series.
        OUTPUTS:
            - loss_val: float
                Patton loss value.
        '''
        loss_val = np.mean(
            np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e)
        ) + self.lambdas['e'] * np.mean( np.where(e>v, e-v, 0) ) +\
            self.lambdas['q'] * np.mean( np.where(v>0, v, 0) )
        return loss_val
    
    def Joint_SAVloop(self, beta, y, q0, e0, pred_mode=False):
        '''
        Loop for the joint ^q and ^e estimation via SAV specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (8,).
            - y: ndarray
                target time series.
            - q0: float
                if pred_mode=False, it contains ^q starting point. Otherwise, it is not used.
            - e0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], e[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile forecast.
            - e: ndarray
                Expected Shortfall forecast.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute q and e at step 0
            q = list()
            e = list()
            q.append( beta[0] + beta[1] * np.abs(e0[0]) + beta[2] * e0[1] ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( beta[3] + beta[4] * np.abs(e0[0]) + beta[5] * e0[2] )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] )
            e.append( beta[3] + beta[4] * np.abs(y[t-1]) + beta[5] * e[t-1] )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def Joint_ASloop(self, beta, y, q0, e0, pred_mode=False):
        '''
        Loop for the joint ^q and ^e estimation via AS specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (10,).
            - y: ndarray
                target time series.
            - q0: float
                if pred_mode=False, it contains ^q starting point. Otherwise, it is not used.
            - e0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], e[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile forecast.
            - e: ndarray
                Expected Shortfall forecast.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute q and e at step 0
            q = list()
            e = list()
            q.append( beta[0] + np.where(e0[0]>0, beta[1], beta[2]) * e0[0] + beta[3] * e0[1] ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( beta[4] + np.where(e0[0]>0, beta[5], beta[6]) * e0[0] + beta[7] * e0[2] )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        y_plus_coeff_q = np.where(y>0, beta[1], beta[2]) #Only one between positive and negative y part gives a contribution
        y_plus_coeff_e = np.where(y>0, beta[5], beta[6])
        for t in range(1, len(y)):
            q.append( beta[0] + y_plus_coeff_q[t-1] * y[t-1] + beta[3] * q[t-1] )
            e.append( beta[4] + y_plus_coeff_e[t-1] * y[t-1] + beta[7] * e[t-1] )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def Joint_GARCHloop(self, beta, y, q0, e0, pred_mode=False):
        '''
        Loop for the joint ^q and ^e estimation via GARCH specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (8,).
            - y: ndarray
                target time series.
            - q0: float
                if pred_mode=False, it contains ^q starting point. Otherwise, it is not used.
            - e0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], e[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile forecast.
            - e: ndarray
                Expected Shortfall forecast.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute q and e at step 0
            q = list()
            e = list()
            q.append( -np.sqrt(beta[0]+beta[1]*e0[0]**2+beta[2]*e0[1]**2) ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( -np.sqrt(beta[3]+beta[4]*e0[0]**2+beta[5]*e0[2]**2) )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2) )
            e.append( -np.sqrt(beta[3]+beta[4]*y[t-1]**2+beta[5]*e[t-1]**2) )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def Jointloss(self, beta, y, q0, e0):
        '''
        Compute the Patton loss of the model.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (2*self.n_parameters,).
            - y: ndarray
                target time series.
            - q0: float
                initial point for the ^q forecast.
            - e0: float
                initial point for the ^e forecast.
        OUTPUTS:
            - loss_val: float
                loss value.
        '''
        q, e = self.joint_loop(beta, y, q0, e0)
        loss_val = self.joint_loss(q, e, y) #Compute loss
        return loss_val
    
    def joint_fit4mp(self, yi, q0, e0, beta0, n_rep, pipend):
        '''
        Joint optimization routine.
        INPUTS:
            - yi: ndarray
                target time series.
            - q0: float
                initial point for the ^q forecast.
            - e0: float
                initial point for the ^e forecast.
            - beta0: ndarray
                initial point for the optimization. The shape is (2*self.n_parameters,).
            - n_rep: int
                number of repetitions for the optimization.
            - pipend: multiprocessing.connection.Connection
                pipe end for communicating multiprocessing.
        OUTPUTS:
            - None.
        '''
        from scipy.optimize import minimize

        # First iteration
        res = minimize(
            lambda x: self.Jointloss(x, yi, q0, e0), beta0,
            method='SLSQP', options={'disp':False})
        beta_worker, fval_beta_worker, exitflag_worker = res.x, res.fun, int(res.success)

        # Iterate until the optimization is successful or the maximum number of repetitions is reached
        for _ in range(n_rep):
            res = minimize(
                lambda x: self.Jointloss(x, yi, q0, e0), beta_worker,
                method='SLSQP', options={'disp':False})
            beta_worker, fval_beta_worker, exitflag_worker = res.x, res.fun, int(res.success)
            #If optimization is successful, exit the loop (no need to iterate further repetitions)
            if exitflag_worker == 1:
                break

        # Communicate the results to the main process
        pipend.send((beta_worker, fval_beta_worker, exitflag_worker))

    def fit(self, yi, seed=None, return_train=False, q0=None, nV=102, n_init=3, n_rep=5):
        '''
        Fit the CAESar model.
        INPUTS:
            - yi: ndarray
                target time series.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                if True, the function returns the fitted values. Default is False.
            - q0: list of float or None, optional
                [initial quantile, initial expected shortfall]. If None, the initial quantile
                is computed as the empirical quantile in the first 10% of the
                training set; the initial expected shortfall as the tail mean in the same
                subset. Default is None.
            - nV: int, optional
                number of random initializations of the model coefficients. Default is 102.
            - n_init: int, optional
                number of best initializations to work with. Default is 3.
            - n_rep: int, optional
                number of repetitions for the optimization. Default is 5.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - ei: ndarray
                expected shortfall forecast in the training set (if return_train=True).
            - beta: ndarray
                optimized coefficients of the model. The shape is
                (2, self.n_parameters) (if return_train=True).
        '''
        #-------------------- Step 0: CAViaR for quantile initial guess
        cav_res = CAViaR(
            self.theta, self.mdl_spec).fit(
                yi, seed=seed, return_train=True, q0=q0) #Train CAViaR
        qi, beta_cav = cav_res['qi'], cav_res['beta']
        del(cav_res)

        #-------------------- Step 1: Initialization
        if isinstance(yi, list):
            yi = np.array(yi)

        if isinstance(q0, type(None)):
            #The starting forecast ^q_0 and ^e_0 is the empricial quantile of the first part of the trainig set (for computational reason)
            n_emp = int(np.ceil(0.1 * len(yi))) #Select onyl 1/10 of the training set
            if round(n_emp * self.theta) == 0: n_emp = len(yi) #In case the training dimension is too small wrt theta
            y_sort = np.sort(yi[:n_emp])
            quantile0 = int(round(n_emp * self.theta))-1
            if quantile0 < 0: quantile0 = 0
            e0 = np.mean(y_sort[:quantile0+1])
            q0 = y_sort[quantile0]
        else:
            q0 = q0[0]
            e0 = q0[1]

        #-------------------- Step 2: Initial guess
        np.random.seed(seed)
        nInitialVectors = [nV//3, self.n_parameters] #Define the shape of the random initializations
        beta0 = [np.random.uniform(0, 1, nInitialVectors)]
        beta0.append(np.random.uniform(-1, 1, nInitialVectors))
        beta0.append(np.random.randn(*nInitialVectors))
        beta0 = np.concatenate(beta0, axis=0)
        beta0 = np.concatenate([np.concatenate([beta_cav.reshape(1,-1)]*nV,
                                               axis=0), beta0], axis=1)
        AEfval = np.empty(nV) #Initialize the loss function vector
        #Iterates over the random initializations
        for i in range(nV):
            AEfval[i] = self.Jointloss(beta0[i, :], yi, q0, e0) #Compute starting loss
        beta0 = beta0[AEfval.argsort()][0:n_init] #Sort initializations by loss and select only the n_init best initializations
        
        #-------------------- Step 3: Optimization Routine
        beta = np.empty((n_init, 2*self.n_parameters)) #Initialize the parameters vector
        fval_beta = np.empty(n_init) #Initialize the loss function vector
        exitflag = np.empty(n_init) #Initialize the exit flag vector
        
        # Multiprocessing: create and start worker processes
        workers = list()
        for i in range(n_init):
            parent_pipend, child_pipend = mp.Pipe() # Create a pipe to communicate with the worker
            worker = mp.Process(target=self.joint_fit4mp,
                                args=(yi, q0, e0, beta0[i, :],
                                      n_rep, child_pipend))
            workers.append([worker, parent_pipend])
            worker.start()

        # Gather results from workers
        for i, worker_list in enumerate(workers):
            worker, parent_pipend = worker_list
            beta_worker, fval_beta_worker, exitflag_worker =\
                parent_pipend.recv() # Get the result from the worker
            worker.join()
            beta[i, :] = beta_worker
            fval_beta[i] = fval_beta_worker
            exitflag[i] = exitflag_worker
        
        ind_min = np.argmin(fval_beta) #Select the index of the best loss
        self.beta = beta[ind_min, :] #Store the best parameters
        self.fval_beta = fval_beta[ind_min] #Store the best loss
        self.exitflag = exitflag[ind_min] #Store the best exit flag

        #Compute the fit output: optimal beta vector, optimization info, and the last pair ^q, ^e
        #in sample variables
        qi, ei = self.joint_loop(self.beta, yi, q0, e0)
        self.last_state = [yi[-1], qi[-1], ei[-1]]
        if return_train:
            return {'qi':qi, 'ei':ei, 'beta':self.beta.reshape((2, self.n_parameters))}
    
    def predict(self, yf=np.array(list())):
        '''
        Predict the quantile.
        INPUTS:
            - yf: ndarray, optional
                test data. If yf is not empty, the internal state is updated
                with the last observation. Default is an empty list.
        OUTPUTS:
            - qf: ndarray
                quantile estimate.
            - ef: ndarray
                Expected Shortfall estimate.
        '''
        qf, ef = self.joint_loop(self.beta, yf, None, self.last_state, pred_mode=True)
        if len(yf) > 0:
            self.last_state = [yf[-1], qf[-1], ef[-1]]
        return {'qf':qf, 'ef':ef}

    def fit_predict(self, y, ti, seed=None, return_train=True, q0=None, nV=102, n_init=3, n_rep=5):
        '''
        Fit and predict the CAESar model.
        INPUTS:
            - y: ndarray
                target time series.
            - ti: int
                train set length.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                return the train set. Default is True.
            - q0: list of float or None, optional
                [initial quantile, initial expected shortfall]. If None, the initial quantile
                is computed as the empirical quantile in the first 10% of the
                training set; the initial expected shortfall as the tail mean in the same
                subset. Default is None.
            - nV: int, optional
                number of random initializations of the model coefficients. Default is 102.
            - n_init: int, optional
                number of best initializations to work with. Default is 3.
            - n_rep: int, optional
                number of repetitions for the optimization. Default is 5.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - ei: ndarray
                expected shortfall forecast in the training set (if return_train=True).
            - qf: ndarray
                quantile forecast in the test set.
            - ef: ndarray
                expected shortfall forecast in the test set.
            - beta: ndarray
                optimized coefficients of the model. The shape is (2, self.n_parameters).
        '''
        yi, yf = y[:ti], y[ti:] #Split train and test
        if return_train:
            res_train = self.fit(yi, seed=seed, return_train=True, q0=q0) #Train AE
            res_test = self.predict(yf)
            return {'qi':res_train['qi'], 'ei':res_train['ei'], 'qf':res_test['qf'], 'ef':res_test['ef'],
                    'beta':self.beta.reshape((2, self.n_parameters))} #Return prediction
        else:
            self.fit(yi, seed=seed, return_train=False, q0=q0) #Train AE
            res_test = self.predict(yf)
            return {'qf':res_test['qf'], 'ef':res_test['ef'],
                    'beta':self.beta.reshape((2, self.n_parameters))} #Return prediction
