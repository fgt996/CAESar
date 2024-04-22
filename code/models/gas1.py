
import numpy as np

class GAS1():
    def __init__(self, theta):
        self.theta = theta

    def loss(self, v, e, y):
        return np.mean(
            np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e)
        )
    
    def smooth_loss(self, v, e, y, tau):
        return np.mean(
            (y-v)/( (1 + np.exp(tau*(y-v)))*self.theta*e ) + v/e + np.log(-e)
        )
            
    def GAS1_loop(self, beta, y, k0, pred_mode=False):
        # Initial point
        if pred_mode: #If we are in prediction mode, we have k at step -1 and we need to compute k at step 0
            k = list()
            k.append(
                beta[2]*k0[1] + beta[3]*(np.where(
                    k0[0]<=beta[0]*k0[1], k0[0]/self.theta, 0) - beta[1]*k0[1]) / beta[1]*k0[1]
                ) #In pred_mode, k0 is assumed to be [y[-1], k[-1]]
            q, e = [beta[0]*np.exp(k[0])], [beta[1]*np.exp(k[0])]
        else: #If we are in training mode, we have an approximation of k at step 0
            k = [k0]
            q, e = [beta[0]*np.exp(k0)], [beta[1]*np.exp(k0)]
        
        # Loop
        for t in range(1, len(y)):
            k.append(beta[2]*k[t-1] +\
                     beta[3]*(np.where(y[t-1]<=q[t-1], y[t-1]/self.theta, 0) - e[t-1]) / e[t-1])
            q.append(beta[0]*np.exp(k[t]))
            e.append(beta[1]*np.exp(k[t]))
        return np.array(q), np.array(e), np.array(k)
    
    def GASloss(self, beta, y, point0, tau=None):
        q, e, _ = self.GAS1_loop(beta, y, point0)
        if isinstance(tau, type(None)):
            return self.loss(q, e, y)
        else:
            return self.smooth_loss(q, e, y, tau)
    
    def fit_core(self, yi, beta0, n_rep, k0, tau=None):
        from scipy.optimize import minimize
        if isinstance(tau, type(None)):
            bound = [(None,0), (None,0), (None,None), (None,None)]
            constraints = [{'type': 'ineq', 'fun': lambda x: x[0]-x[1]}]
            res = minimize(lambda x: self.GASloss(x, yi, k0, tau), beta0, bounds=bound,
                            constraints=constraints, method='SLSQP', options={'disp': False})
            self.opt_res = res
            self.tau = tau
            if res.status == 0:
                return res.x
            else:
                return beta0
        else:
            bound = [(None,0), (None,0), (None,None), (None,None)]
            constraints = [{'type': 'ineq', 'fun': lambda x: x[0]-x[1]}]
            res = minimize(lambda x: self.GASloss(x, yi, k0, tau), beta0, bounds=bound,
                            constraints=constraints, method='SLSQP', options={'disp': False})
            self.opt_res = res
            self.tau = tau
            if res.status == 0:
                return res.x
            else:
                return beta0

    def fit(self, yi, seed=None, return_train=False):
        import multiprocessing as mp
        from scipy.optimize import fmin
        
        #-------------------- Step 1: Initialization
        if isinstance(yi, list):
            yi = np.array(yi)
        n_rep = 5 #Set the number of repetitions

        #The starting forecast ^q_0 and ^e_0 is the empricial quantile of the first part of the trainig set (for computational reason)
        n_emp = int(np.ceil(0.1 * len(yi))) #Select onyl 1/10 of the training set
        if round(n_emp * self.theta) == 0: n_emp = len(yi) #In case the training dimension is too small wrt theta
        y_sort = np.sort(yi[:n_emp])
        quantile0 = int(round(n_emp * self.theta))-1
        if quantile0 < 0: quantile0 = 0
        k0 = np.log(-y_sort[quantile0]) if y_sort[quantile0]<0 else np.log(y_sort[quantile0])
        self.beta0 = np.array([-1.164, -1.757, 0.995, 0.007])

        #-------------------- Step 2: Optimization Routine
        np.random.seed(seed)
        # First optimization: tau = 5
        self.beta = self.fit_core(yi, self.beta0, n_rep, k0, tau=5)
        # Second optimization: tau = 20
        self.beta = self.fit_core(yi, self.beta, n_rep, k0, tau=20)
        # Second optimization: actual loss
        self.beta = self.fit_core(yi, self.beta, n_rep, k0, tau=None)
        
        # Save in sample variables
        qi, ei, ki = self.GAS1_loop(self.beta, yi, k0)
        self.train_out = {'qi':qi, 'ei':ei, 'ki':ki}
        self.last_state = [qi[-1], ki[-1]]
        if return_train:
            return {'qi':qi, 'ei':ei, 'beta':self.beta}
    
    def predict(self, yf):
        qf, ef, kf = self.GAS1_loop(self.beta, yf, self.last_state, pred_mode=True)
        self.last_state = [qf[-1], kf[-1]]
        return qf, ef

    def fit_predict(self, y, ti, seed=None, return_train=False):
        yi, yf = y[:ti], y[ti:] #Split train and test
        if return_train:
            res_train = self.fit(yi, seed=seed, return_train=True) #Train AE
            qf, ef = self.predict(yf)
            return {'qi':res_train['qi'], 'ei':res_train['ei'],
                    'qf':qf, 'ef':ef, 'beta':self.beta} #Return prediction
        else:
            self.fit(yi, seed=seed, return_train=False) #Train AE
            qf, ef = self.predict(yf)
            return {'qf':qf, 'ef':ef, 'beta':self.beta} #Return prediction
        