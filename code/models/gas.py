
import numpy as np

class GAS1():
    '''
    GAS1 model for Expected Shortfall estimation.
    '''
    def __init__(self, theta):
        '''
        Initialization of the GAS1 model.
        INPUTS:
            - theta: float
                desired confidence level.
        OUTPUTS:
            - None
        '''
        self.theta = theta

    def loss(self, v, e, y):
        '''
        Compute the GAS1 loss.
        INPUTS:
            - v: float
                quantile forecast.
            - e: float
                expected shortfall forecast.
            - y: ndarray
                target time series.
        OUTPUTS:
            - loss_val: float
                GAS1 loss.
        '''
        loss_val = np.mean(
            np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e)
        )
        return loss_val
    
    def smooth_loss(self, v, e, y, tau):
        '''
        Compute the smooth version of the GAS1 loss.
        INPUTS:
            - v: float
                quantile forecast.
            - e: float
                expected shortfall forecast.
            - y: ndarray
                target time series.
            - tau: float
                smoothing parameter.
        OUTPUTS:
            - loss_val: float
                GAS1 loss.
        '''
        loss_val = np.mean(
            (y-v)/( (1 + np.exp(tau*(y-v)))*self.theta*e ) + v/e + np.log(-e)
        )
        return loss_val
            
    def GAS1_loop(self, beta, y, k0, pred_mode=False):
        '''
        Loop for the GAS1 model.
        INPUTS:
            - beta: ndarray
                model parameters.
            - y: ndarray
                target time series.
            - k0: float
                initial point for the state variable.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile forecast.
            - e: ndarray
                expected shortfall forecast.
            - k: ndarray
                state variable.
        '''
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
        q, e, k = np.array(q), np.array(e), np.array(k)
        return q, e, k
    
    def GASloss(self, beta, y, k0, tau=None):
        '''
        Compute the GAS1 loss.
        INPUTS:
            - beta: ndarray
                model parameters.
            - y: ndarray
                target time series.
            - k0: float
                initial point for the state variable.
            - tau: float, optional
                smoothing parameter. If None, the original loss is computed. Default is None.
        OUTPUTS:
            - loss_val: float
                GAS1 loss, either in its original form (when tau=None) or smoothed version.
        '''
        q, e, _ = self.GAS1_loop(beta, y, k0)
        if isinstance(tau, type(None)):
            loss_val = self.loss(q, e, y)
        else:
            loss_val = self.smooth_loss(q, e, y, tau)
        return loss_val
    
    def fit_core(self, yi, beta0, k0, tau=None):
        '''
        Core function for the GAS1 model.
        INPUTS:
            - yi: ndarray
                target time series.
            - beta0: ndarray
                initial model parameters.
            - k0: float 
                initial point for the state variable.
            - tau: float, optional
                smoothing parameter. If None, the original loss is computed. Default is None.
        OUTPUTS:
            - beta: ndarray
                optimized model parameters if the optimization is successful,
                otherwise the initial parameters.
        '''
        from scipy.optimize import minimize
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
        '''
        Fit the GAS1 model.
        INPUTS:
            - yi: ndarray
                target time series.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                if True, return the fitted variables. Default is False.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - ei: ndarray
                expected shortfall in the training set (if return_train=True).
            - ki: ndarray
                state variable in the training set (if return_train=True).
            - beta: ndarray
                optimized model parameters (if return_train=True).
        '''
        import multiprocessing as mp
        from scipy.optimize import fmin
        
        #-------------------- Step 1: Initialization
        if isinstance(yi, list):
            yi = np.array(yi)

        #The starting forecast ^q_0 and ^e_0 is the empricial quantile of the first part of the trainig set (for computational reason)
        n_emp = int(np.ceil(0.1 * len(yi))) #Select onyl 1/10 of the training set
        if round(n_emp * self.theta) == 0: n_emp = len(yi) #In case the training dimension is too small wrt theta
        y_sort = np.sort(yi[:n_emp])
        quantile0 = int(round(n_emp * self.theta))-1
        if quantile0 < 0: quantile0 = 0
        k0 = np.log(-y_sort[quantile0]) if y_sort[quantile0]<0 else np.log(y_sort[quantile0])
        # The initial guess for model coefficients is taken from the original paper
        self.beta0 = np.array([-1.164, -1.757, 0.995, 0.007])

        #-------------------- Step 2: Optimization Routine
        np.random.seed(seed)
        # First optimization: tau = 5
        self.beta = self.fit_core(yi, self.beta0, k0, tau=5)
        # Second optimization: tau = 20
        self.beta = self.fit_core(yi, self.beta, k0, tau=20)
        # Second optimization: actual loss
        self.beta = self.fit_core(yi, self.beta, k0, tau=None)
        
        # Save in sample variables
        qi, ei, ki = self.GAS1_loop(self.beta, yi, k0)
        self.train_out = {'qi':qi, 'ei':ei, 'ki':ki}
        self.last_state = [yi[-1], ki[-1]]
        if return_train:
            return {'qi':qi, 'ei':ei, 'ki':ki, 'beta':self.beta}
    
    def predict(self, yf=np.array(list())):
        '''
        Predict the GAS1 model.
        INPUTS:
            - yf: ndarray, optional
                target time series. If yf is not empty, the internal state is updated
                with the last observation. Default is an empty list.
        OUTPUTS:
            - qf: ndarray
                quantile forecast in the test set.
            - ef: ndarray
                expected shortfall forecast in the test set.
            - kf: ndarray
                state variable in the test set.
        '''
        qf, ef, kf = self.GAS1_loop(self.beta, yf, self.last_state, pred_mode=True)
        if len(yf) > 0:
            self.last_state = [yf[-1], kf[-1]]
        return {'qf':qf, 'ef':ef, 'kf':kf}

    def fit_predict(self, y, ti, seed=None, return_train=False):
        '''
        Fit and predict the GAS1 model.
        INPUTS:
            - y: ndarray
                target time series.
            - ti: int
                train set length.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                if True, return the fitted values in training set. Default is False.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - ei: ndarray
                expected shortfall forecast in the training set (if return_train=True).
            - ki: ndarray
                state variable in the training set (if return_train=True).
            - qf: ndarray
                quantile forecast in the test set.
            - ef: ndarray
                expected shortfall forecast in the test set.
            - kf: ndarray
                state variable in the test set.
            - beta: ndarray
                optimized model parameters.
        '''
        yi, yf = y[:ti], y[ti:] #Split train and test
        if return_train:
            res_train = self.fit(yi, seed=seed, return_train=True) #Train AE
            res_test = self.predict(yf)
            return {'qi':res_train['qi'], 'ei':res_train['ei'], 'ki':res_train['ki'],
                    'qf':res_test['qf'], 'ef':res_test['ef'], 'kf':res_test['kf'],
                    'beta':self.beta} #Return prediction
        else:
            self.fit(yi, seed=seed, return_train=False) #Train AE
            res_test = self.predict(yf)
            return {'qf':res_test['qf'], 'ef':res_test['ef'], 'kf':res_test['kf'],
                    'beta':self.beta} #Return prediction
        
class GAS2():
    '''
    GAS2 model for Expected Shortfall estimation.
    '''
    def __init__(self, theta):
        '''
        Initialization of the GAS2 model.
        INPUTS:
            - theta: float
                desired confidence level.
        OUTPUTS:
            - None
        '''
        self.theta = theta

    def loss(self, v, e, y):
        '''
        Compute the GAS2 loss.
        INPUTS:
            - v: float
                quantile forecast.
            - e: float
                expected shortfall forecast.
            - y: ndarray
                target time series.
        OUTPUTS:
            - loss_val: float
                GAS2 loss.
        '''
        loss_val = np.mean(
            np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e)
        )
        return loss_val
    
    def smooth_loss(self, v, e, y, tau):
        '''
        Compute the smooth version of the GAS2 loss.
        INPUTS:
            - v: float
                quantile forecast.
            - e: float
                expected shortfall forecast.
            - y: ndarray
                target time series.
            - tau: float
                smoothing parameter.
        OUTPUTS:
            - loss_val: float
                GAS2 loss.
        '''
        loss_val = np.mean(
            (y-v)/( (1 + np.exp(tau*(y-v)))*self.theta*e ) + v/e + np.log(-e)
        )
        return loss_val
            
    def GAS2_loop(self, beta, y, k0, pred_mode=False):
        '''
        Loop for the GAS2 model.
        INPUTS:
            - beta: ndarray
                model parameters.
            - y: ndarray
                target time series.
            - k0: float
                initial point for the state variable.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile forecast.
            - e: ndarray
                expected shortfall forecast.
            - k: ndarray
                state variable.
        '''
        lam = np.zeros(2) # Initialize an array useful for calculations
        # Initial point
        if pred_mode: #If we are in pred_mode, we have q and e at step -1 and we need to compute them at step 0
            lam[0] = -k0[1]*( (k0[0]<=k0[1]) - self.theta ) #In pred_mode, k0 is assumed to be [y[-1], q[-1], k[-1]]
            lam[1] = k0[0]*(k0[0]<=k0[1])/self.theta - k0[2]
            qe_temp = beta[:2] + np.dot(np.eye(2)*beta[2:4], np.array(k0[1:])) +\
                np.dot(beta[4:].reshape(2,2), lam)
            q = [qe_temp[0]]; e = [qe_temp[1]]
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q, e = [k0[0]], [k0[1]]
            qe_temp = np.array(k0)
            
        # Loop
        for t in range(1, len(y)):
            lam[0] = -q[t-1]*( (y[t-1]<=q[t-1]) - self.theta )
            lam[1] = y[t-1]*(y[t-1]<=q[t-1])/self.theta - e[t-1]
            qe_temp = beta[:2] + np.dot(np.eye(2)*beta[2:4], qe_temp) +\
                np.dot(beta[4:].reshape(2,2), lam)
            q.append(qe_temp[0]); e.append(qe_temp[1])
        q, e = np.array(q), np.array(e)
        return q, e
    
    def GASloss(self, beta, y, k0, tau=None):
        '''
        Compute the GAS2 loss.
        INPUTS:
            - beta: ndarray
                model parameters.
            - y: ndarray
                target time series.
            - k0: float
                initial point for the state variable.
            - tau: float, optional
                smoothing parameter. If None, the original loss is computed. Default is None.
        OUTPUTS:
            - loss_val: float
                GAS2 loss, either in its original form (when tau=None) or smoothed version.
        '''
        q, e  = self.GAS2_loop(beta, y, k0)
        if isinstance(tau, type(None)):
            loss_val = self.loss(q, e, y)
        else:
            loss_val = self.smooth_loss(q, e, y, tau)
        return loss_val
    
    def fit_core(self, yi, beta0, k0, tau=None):
        '''
        Core function for the GAS2 model.
        INPUTS:
            - yi: ndarray
                target time series.
            - beta0: ndarray
                initial model parameters.
            - k0: float 
                initial point for the state variable.
            - tau: float, optional
                smoothing parameter. If None, the original loss is computed. Default is None.
        OUTPUTS:
            - beta: ndarray
                optimized model parameters if the optimization is successful,
                otherwise the initial parameters.
        '''
        from scipy.optimize import minimize
        bound = [(None,0), (None,0), (0,None), (0,None),
                 (None,0), (None,0), (None,0), (None,0)]
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
        '''
        Fit the GAS2 model.
        INPUTS:
            - yi: ndarray
                target time series.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                if True, return the fitted variables. Default is False.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - ei: ndarray
                expected shortfall in the training set (if return_train=True).
            - beta: ndarray
                optimized model parameters (if return_train=True).
        '''
        import multiprocessing as mp
        from scipy.optimize import fmin
        
        #-------------------- Step 1: Initialization
        if isinstance(yi, list):
            yi = np.array(yi)

        #The starting forecast ^q_0 and ^e_0 is the empricial quantile of the first part of the trainig set (for computational reason)
        n_emp = int(np.ceil(0.1 * len(yi))) #Select onyl 1/10 of the training set
        if round(n_emp * self.theta) == 0: n_emp = len(yi) #In case the training dimension is too small wrt theta
        y_sort = np.sort(yi[:n_emp])
        quantile0 = int(round(n_emp * self.theta))-1
        if quantile0 < 0: quantile0 = 0
        k0 = np.array([y_sort[quantile0], np.mean(y_sort[:quantile0+1])]) #Initial state variable
        # The initial guess for model coefficients is taken from the original paper
        self.beta0 = np.array([-0.009, -0.010, 0.993, 0.994, -0.358, -0.003, -0.351, -0.003])

        #-------------------- Step 2: Optimization Routine
        np.random.seed(seed)
        # First optimization: tau = 5
        self.beta = self.fit_core(yi, self.beta0, k0, tau=5)
        # Second optimization: tau = 20
        self.beta = self.fit_core(yi, self.beta, k0, tau=20)
        # Second optimization: actual loss
        self.beta = self.fit_core(yi, self.beta, k0, tau=None)
        
        # Save in sample variables
        qi, ei = self.GAS2_loop(self.beta, yi, k0)
        self.train_out = {'qi':qi, 'ei':ei}
        self.last_state = [yi[-1], qi[-1], ei[-1]]
        if return_train:
            return {'qi':qi, 'ei':ei, 'beta':self.beta}
    
    def predict(self, yf=np.array(list())):
        '''
        Predict the GAS2 model.
        INPUTS:
            - yf: ndarray, optional
                target time series. If yf is not empty, the internal state is updated
                with the last observation. Default is an empty list.
        OUTPUTS:
            - qf: ndarray
                quantile forecast in the test set.
            - ef: ndarray
                expected shortfall forecast in the test set.
        '''
        qf, ef = self.GAS2_loop(self.beta, yf, self.last_state, pred_mode=True)
        if len(yf) > 0:
            self.last_state = [yf[-1], qf[-1], ef[-1]]
        return {'qf':qf, 'ef':ef}

    def fit_predict(self, y, ti, seed=None, return_train=False):
        '''
        Fit and predict the GAS1 model.
        INPUTS:
            - y: ndarray
                target time series.
            - ti: int
                train set length.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                if True, return the fitted values in training set. Default is False.
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
                optimized model parameters.
        '''
        yi, yf = y[:ti], y[ti:] #Split train and test
        if return_train:
            res_train = self.fit(yi, seed=seed, return_train=True) #Train AE
            res_test = self.predict(yf)
            return {'qi':res_train['qi'], 'ei':res_train['ei'], 'qf':res_test['qf'],
                    'ef':res_test['ef'], 'beta':self.beta} #Return prediction
        else:
            self.fit(yi, seed=seed, return_train=False) #Train AE
            res_test = self.predict(yf)
            return {'qf':res_test['qf'], 'ef':res_test['ef'], 'beta':self.beta} #Return prediction
