
import numpy as np
import multiprocessing as mp
from models.caviar import CAViaR

class B_CAESar():
    def __init__(self, theta, spec='AS', lambdas=dict()):
        self.theta = theta #Initialize theta
        self.lambdas = {'r':10}
        self.lambdas.update(lambdas) #Initialize lambdas for soft constraints

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 4
            self.loop = self.R_SAVloop
            self.joint_loop = self.Joint_SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 5
            self.loop = self.R_ASloop
            self.joint_loop = self.Joint_ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 4
            self.loop = self.R_GARCHloop
            self.joint_loop = self.Joint_GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')
    
    def loss_function(self, v, r, y):
        return np.mean(
            (r - np.where(y<v, (y-v)/self.theta, 0))**2
            ) + self.lambdas['r'] * np.mean( np.where(r>0, r, 0) )
    
    def R_SAVloop(self, beta, y, q, r0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and r at step -1 and we need to compute r at step 0
            r = list()
            r.append( beta[0] + beta[1] * np.abs(r0[0]) + beta[2] * r0[1] + beta[3] * r0[2] ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of r at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * r[t-1] )
        return np.array(r)
    
    def Joint_SAVloop(self, beta, y, q0, e0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute q and e at step 0
            q = list()
            e = list()
            q.append( beta[0] + beta[1] * np.abs(e0[0]) + beta[2] * e0[1] + beta[3] * e0[2] ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( beta[4] + beta[5] * np.abs(e0[0]) + beta[6] * e0[1] + beta[7] * e0[2] )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * e[t-1] )
            e.append( beta[4] + beta[5] * np.abs(y[t-1]) + beta[6] * q[t-1] + beta[7] * e[t-1] )
        return np.array(q), np.array(e)
    
    def R_ASloop(self, beta, y, q, r0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute e at step 0
            r = list()
            r.append( beta[0] + np.where(r0[0]>0, beta[1], beta[2]) * r0[0] + beta[3] * r0[1] + beta[4] * r0[2] ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of e at step 0
            r = [r0]

        # Loop
        y_plus_coeff = np.where(y>0, beta[1], beta[2]) #Only one between positive and negative y part gives a contribution
        for t in range(1, len(y)):
            r.append( beta[0] + y_plus_coeff[t-1] * y[t-1] + beta[3] * q[t-1] + beta[4] * r[t-1] )
        return np.array(r)
    
    def Joint_ASloop(self, beta, y, q0, e0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute q and e at step 0
            q = list()
            e = list()
            q.append( beta[0] + np.where(e0[0]>0, beta[1], beta[2]) * e0[0] + beta[3] * e0[1] + beta[4] * e0[2] ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( beta[5] + np.where(e0[0]>0, beta[6], beta[7]) * e0[0] + beta[8] * e0[1] + beta[9] * e0[2] )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        y_plus_coeff_q = np.where(y>0, beta[1], beta[2]) #Only one between positive and negative y part gives a contribution
        y_plus_coeff_e = np.where(y>0, beta[6], beta[7])
        for t in range(1, len(y)):
            q.append( beta[0] + y_plus_coeff_q[t-1] * y[t-1] + beta[3] * q[t-1] + beta[4] * e[t-1] )
            e.append( beta[5] + y_plus_coeff_e[t-1] * y[t-1] + beta[8] * q[t-1] + beta[9] * e[t-1] )
        return np.array(q), np.array(e)
    
    def R_GARCHloop(self, beta, y, q, r0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute e at step 0
            r = list()
            r.append( -np.sqrt(beta[0]+beta[1]*r0[0]**2+beta[2]*r0[1]**2+beta[3]*r0[2]**2) ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of e at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*r[t-1]**2) )
        return np.array(r)
    
    def Joint_GARCHloop(self, beta, y, q0, e0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute q and e at step 0
            q = list()
            e = list()
            q.append( -np.sqrt(beta[0]+beta[1]*e0[0]**2+beta[2]*e0[1]**2+beta[3]*e0[2]**2) ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( -np.sqrt(beta[4]+beta[5]*e0[0]**2+beta[6]*e0[1]**2+beta[7]*e0[2]**2) )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*e[t-1]**2) )
            e.append( -np.sqrt(beta[4]+beta[5]*y[t-1]**2+beta[6]*q[t-1]**2+beta[7]*e[t-1]**2) )
        return np.array(q), np.array(e)

    def ESloss(self, beta, y, q, e0):
        r = self.loop(beta, y, q, e0)
        return self.loss_function(q, r, y)
    
    def optim4mp(self, yi, qi, r0, beta0, n_rep, output_queue):
        from scipy.optimize import fmin

        # First iteration
        beta_worker, fval_beta_worker, _, _, temp = fmin(
            lambda x: self.ESloss(x, yi, qi, r0), beta0,
            disp=False,full_output=True)
        exitflag_worker = 1 if temp==0 else 0

        # Iterate until the optimization is successful or the maximum number of repetitions is reached
        for _ in range(n_rep):
            beta_worker, fval_beta_worker, _, _, temp = fmin(
                lambda x: self.ESloss(x, yi, qi, r0),
                beta_worker, disp=False, full_output=True) #Minimize over beta
            exitflag_worker = 1 if temp==0 else 0
            #If optimization is successful, exit the loop (no need to iterate further repetitions)
            if exitflag_worker == 1:
                break

        # Put results in output queue
        output_queue.put((beta_worker, fval_beta_worker, exitflag_worker))

    def fit(self, yi, seed=None, return_train=False, q0=None):
        #-------------------- Step 0: CAViaR for quantile initial guess
        cav_res = CAViaR(
            self.theta, self.mdl_spec).fit(
                yi, seed=seed, return_train=True, q0=q0) #Train CAViaR
        qi, beta_cav = cav_res['qi'], cav_res['beta']
        del(cav_res)

        #-------------------- Step 1: Initialization
        if isinstance(yi, list):
            yi = np.array(yi)
        nV, nC, n_rep = 102, 3, 5 #Set the number of: initial vectors; initial parameters; repetitions

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
        AEfval = np.empty(nV) #Initialize the loss function vector
        #Iterates over the random initializations
        for i in range(nV):
            AEfval[i] = self.ESloss(beta0[i, :], yi, qi, e0-q0) #Compute starting loss
        beta0 = beta0[AEfval.argsort()][0:nC] #Sort initializations by loss and select only the nC best initializations
        
        #-------------------- Step 3: Optimization Routine - Only Barrera
        beta = np.empty((nC, self.n_parameters)) #Initialize the parameters vector
        fval_beta = np.empty(nC) #Initialize the loss function vector
        exitflag = np.empty(nC) #Initialize the exit flag vector
        # Prepare the multiprocessing
        output_queue = mp.Queue()
        # Create and start worker processes
        workers = list()
        for i in range(nC):
            worker = mp.Process(target=self.optim4mp,
                                args=(yi, qi, e0-q0, beta0[i, :],
                                      n_rep, output_queue))
            workers.append(worker)
            worker.start()
        # Gather results from workers
        for i, worker in enumerate(workers):
            worker.join()
            beta_worker, fval_beta_worker, exitflag_worker = output_queue.get()
            beta[i, :] = beta_worker
            fval_beta[i] = fval_beta_worker
            exitflag[i] = exitflag_worker
        
        ind_min = np.argmin(fval_beta) #Select the index of the best loss
        self.beta = beta[ind_min, :] #Store the best parameters
        self.fval_beta = fval_beta[ind_min] #Store the best loss
        self.exitflag = exitflag[ind_min] #Store the best exit flag
        
        # Concatenate ^q and ^r parameters, then convert ^r into ^e parameters
        joint_beta = np.concatenate([beta_cav, [0], self.beta]) #Concatenate
        if self.mdl_spec == 'SAV':
            joint_beta[4] += joint_beta[0]
            joint_beta[5] += joint_beta[1]
            joint_beta[6] += joint_beta[2] - joint_beta[7] 
        elif self.mdl_spec == 'AS':
            joint_beta[5] += joint_beta[0]
            joint_beta[6] += joint_beta[1]
            joint_beta[7] += joint_beta[2]
            joint_beta[8] += joint_beta[3] - joint_beta[9]
        elif self.mdl_spec == 'GARCH':
            joint_beta[4] += joint_beta[0]
            joint_beta[5] += joint_beta[1]
            joint_beta[6] += joint_beta[2] + joint_beta[7]
        self.beta = joint_beta

        #Compute the fit output: optimal beta vector, optimization info, and the last pair ^q, ^e
        #in sample variables
        qi, ei = self.joint_loop(self.beta, yi, q0, e0)
        self.last_state = [yi[-1], qi[-1], ei[-1]]
        if return_train:
            return {'qi':qi, 'ei':ei, 'beta':self.beta.reshape((2, self.n_parameters))}
    
    def predict(self, yf):
        qf, ef = self.joint_loop(self.beta, yf, None, self.last_state, pred_mode=True)
        self.last_state = [yf[-1], qf[-1], ef[-1]]
        return qf, ef

    def fit_predict(self, y, ti, seed=None, return_train=True, q0=None):
        yi, yf = y[:ti], y[ti:] #Split train and test
        if return_train:
            res_train = self.fit(yi, seed=seed, return_train=True, q0=q0) #Train AE
            qf, ef = self.predict(yf)
            return {'qi':res_train['qi'], 'ei':res_train['ei'], 'qf':qf, 'ef':ef, 'beta':self.beta.reshape((2, self.n_parameters))} #Return prediction
        else:
            self.fit(yi, seed=seed, return_train=False, q0=q0) #Train AE
            qf, ef = self.predict(yf)
            return {'qf':qf, 'ef':ef, 'beta':self.beta.reshape((2, self.n_parameters))} #Return prediction
    
class P_CAESar():
    def __init__(self, theta, spec='AS', lambdas=dict()):
        self.theta = theta #Initialize theta
        self.lambdas = {'q':10, 'e':10}
        self.lambdas.update(lambdas) #Initialize lambdas for soft constraints

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 4
            self.loop = self.R_SAVloop
            self.joint_loop = self.Joint_SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 5
            self.loop = self.R_ASloop
            self.joint_loop = self.Joint_ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 4
            self.loop = self.R_GARCHloop
            self.joint_loop = self.Joint_GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')
    
    def loss_function(self, v, r, y):
        return np.mean(
            (r - np.where(y<v, (y-v)/self.theta, 0))**2
            ) + self.lambdas['r'] * np.mean( np.where(r>0, r, 0) )
    
    def joint_loss(self, v, e, y):
        return np.mean(
            np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e)
        ) + self.lambdas['e'] * np.mean( np.where(e>v, e-v, 0) ) + self.lambdas['q'] * np.mean( np.where(v>0, v, 0) )
    
    def R_SAVloop(self, beta, y, q, r0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and r at step -1 and we need to compute r at step 0
            r = list()
            r.append( beta[0] + beta[1] * np.abs(r0[0]) + beta[2] * r0[1] + beta[3] * r0[2] ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of r at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * r[t-1] )
        return np.array(r)
    
    def Joint_SAVloop(self, beta, y, q0, e0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute q and e at step 0
            q = list()
            e = list()
            q.append( beta[0] + beta[1] * np.abs(e0[0]) + beta[2] * e0[1] + beta[3] * e0[2] ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( beta[4] + beta[5] * np.abs(e0[0]) + beta[6] * e0[1] + beta[7] * e0[2] )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * e[t-1] )
            e.append( beta[4] + beta[5] * np.abs(y[t-1]) + beta[6] * q[t-1] + beta[7] * e[t-1] )
        return np.array(q), np.array(e)
    
    def R_ASloop(self, beta, y, q, r0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute e at step 0
            r = list()
            r.append( beta[0] + np.where(r0[0]>0, beta[1], beta[2]) * r0[0] + beta[3] * r0[1] + beta[4] * r0[2] ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of e at step 0
            r = [r0]

        # Loop
        y_plus_coeff = np.where(y>0, beta[1], beta[2]) #Only one between positive and negative y part gives a contribution
        for t in range(1, len(y)):
            r.append( beta[0] + y_plus_coeff[t-1] * y[t-1] + beta[3] * q[t-1] + beta[4] * r[t-1] )
        return np.array(r)
    
    def Joint_ASloop(self, beta, y, q0, e0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute q and e at step 0
            q = list()
            e = list()
            q.append( beta[0] + np.where(e0[0]>0, beta[1], beta[2]) * e0[0] + beta[3] * e0[1] + beta[4] * e0[2] ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( beta[5] + np.where(e0[0]>0, beta[6], beta[7]) * e0[0] + beta[8] * e0[1] + beta[9] * e0[2] )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        y_plus_coeff_q = np.where(y>0, beta[1], beta[2]) #Only one between positive and negative y part gives a contribution
        y_plus_coeff_e = np.where(y>0, beta[6], beta[7])
        for t in range(1, len(y)):
            q.append( beta[0] + y_plus_coeff_q[t-1] * y[t-1] + beta[3] * q[t-1] + beta[4] * e[t-1] )
            e.append( beta[5] + y_plus_coeff_e[t-1] * y[t-1] + beta[8] * q[t-1] + beta[9] * e[t-1] )
        return np.array(q), np.array(e)
    
    def R_GARCHloop(self, beta, y, q, r0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute e at step 0
            r = list()
            r.append( -np.sqrt(beta[0]+beta[1]*r0[0]**2+beta[2]*r0[1]**2+beta[3]*r0[2]**2) ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of e at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*r[t-1]**2) )
        return np.array(r)
    
    def Joint_GARCHloop(self, beta, y, q0, e0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute q and e at step 0
            q = list()
            e = list()
            q.append( -np.sqrt(beta[0]+beta[1]*e0[0]**2+beta[2]*e0[1]**2+beta[3]*e0[2]**2) ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( -np.sqrt(beta[4]+beta[5]*e0[0]**2+beta[6]*e0[1]**2+beta[7]*e0[2]**2) )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*e[t-1]**2) )
            e.append( -np.sqrt(beta[4]+beta[5]*y[t-1]**2+beta[6]*q[t-1]**2+beta[7]*e[t-1]**2) )
        return np.array(q), np.array(e)

    def ESloss(self, beta, y, q, e0):
        r = self.loop(beta, y, q, e0)
        return self.loss_function(q, r, y)
    
    def Jointloss(self, beta, y, q0, e0):
        q, e = self.joint_loop(beta, y, q0, e0)
        return self.joint_loss(q, e, y)
    
    def joint_fit4mp(self, yi, q0, e0, beta0, n_rep, output_queue):
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

        # Put results in output queue
        output_queue.put((beta_worker, fval_beta_worker, exitflag_worker))

    def fit(self, yi, seed=None, return_train=False, q0=None):
        #-------------------- Step 0: CAViaR for quantile initial guess
        cav_res = CAViaR(
            self.theta, self.mdl_spec).fit(
                yi, seed=seed, return_train=True, q0=q0) #Train CAViaR
        qi, beta_cav = cav_res['qi'], cav_res['beta']
        del(cav_res)

        #-------------------- Step 1: Initialization
        if isinstance(yi, list):
            yi = np.array(yi)
        nV, nC, n_rep = 102, 3, 5 #Set the number of: initial vectors; initial parameters; repetitions

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
                                               axis=0), np.zeros((nV,1)), beta0], axis=1)
        AEfval = np.empty(nV) #Initialize the loss function vector
        #Iterates over the random initializations
        for i in range(nV):
            AEfval[i] = self.Jointloss(beta0[i, :], yi, q0, e0) #Compute starting loss
        beta0 = beta0[AEfval.argsort()][0:nC] #Sort initializations by loss and select only the nC best initializations
        
        #-------------------- Step 3: Optimization Routine - I step (barrera)
        beta = np.empty((nC, 2*self.n_parameters)) #Initialize the parameters vector
        fval_beta = np.empty(nC) #Initialize the loss function vector
        exitflag = np.empty(nC) #Initialize the exit flag vector
        # Prepare the multiprocessing
        output_queue = mp.Queue()
        # Create and start worker processes
        workers = list()
        for i in range(nC):
            worker = mp.Process(target=self.joint_fit4mp,
                                args=(yi, q0, e0, beta0[i, :],
                                      n_rep, output_queue))
            workers.append(worker)
            worker.start()
        # Gather results from workers
        for i, worker in enumerate(workers):
            worker.join()
            beta_worker, fval_beta_worker, exitflag_worker = output_queue.get()
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
    
    def predict(self, yf):
        qf, ef = self.joint_loop(self.beta, yf, None, self.last_state, pred_mode=True)
        self.last_state = [yf[-1], qf[-1], ef[-1]]
        return qf, ef

    def fit_predict(self, y, ti, seed=None, return_train=True, q0=None):
        yi, yf = y[:ti], y[ti:] #Split train and test
        if return_train:
            res_train = self.fit(yi, seed=seed, return_train=True, q0=q0) #Train AE
            qf, ef = self.predict(yf)
            return {'qi':res_train['qi'], 'ei':res_train['ei'], 'qf':qf, 'ef':ef, 'beta':self.beta.reshape((2, self.n_parameters))} #Return prediction
        else:
            self.fit(yi, seed=seed, return_train=False, q0=q0) #Train AE
            qf, ef = self.predict(yf)
            return {'qf':qf, 'ef':ef, 'beta':self.beta.reshape((2, self.n_parameters))} #Return prediction