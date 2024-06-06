
import numpy as np
import multiprocessing as mp
from models.caviar import CAViaR
import warnings

class CAESar_base():
    def __init__(self, theta, lambdas=dict()):
        self.theta = theta #Initialize theta
        self.lambdas = {'r':10, 'q':10, 'e':10}
        self.lambdas.update(lambdas) #Initialize lambdas for soft constraints
    
    def loss_function(self, v, r, y):
        '''
        Compute the loss function (Barrera) of the model.
        INPUTS:
            - v: ndarray
                Value at Risk.
            - r: ndarray
                difference between Expected Shortfall forecast and Value at Risk.
            - y: ndarray
                target time series.
        OUTPUTS:
            - loss_val: float
                Barrera loss value.
        '''
        loss_val = np.mean(
            (r - np.where(y<v, (y-v)/self.theta, 0))**2
            ) + self.lambdas['r'] * np.mean( np.where(r>0, r, 0) )
        return loss_val
    
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

    def ESloss(self, beta, y, q, r0):
        '''
        Compute the Barrera loss of the model.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (self.n_parameters,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float
                initial point for the ^r forecast.
        OUTPUTS:
            - loss_val: float
                loss value.
        '''
        r = self.loop(beta, y, q, r0) #Compute ^r
        loss_val = self.loss_function(q, r, y) #Compute loss
        return loss_val
    
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
        q, e = self.joint_loop(beta, y, q0, e0) #Compute ^q and ^e
        loss_val = self.joint_loss(q, e, y) #Compute loss
        return loss_val
    
    def optim4mp(self, yi, qi, r0, beta0, n_rep, pipend):
        '''
        Optimization routine for multiprocessing.
        INPUTS:
            - yi: ndarray
                target time series.
            - qi: ndarray
                quantile forecast.
            - r0: float
                initial point for the ^r forecast.
            - beta0: ndarray
                initial point for the optimization. The shape is (self.n_parameters,).
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
            lambda x: self.ESloss(x, yi, qi, r0), beta0,
            method='SLSQP', options={'disp':False})
        beta_worker, fval_beta_worker, exitflag_worker = res.x, res.fun, int(res.success)

        # Iterate until the optimization is successful or the maximum number of repetitions is reached
        for _ in range(n_rep):
            res = minimize(
                lambda x: self.ESloss(x, yi, qi, r0), beta_worker,
                method='SLSQP', options={'disp':False})
            beta_worker, fval_beta_worker, exitflag_worker = res.x, res.fun, int(res.success)
            #If optimization is successful, exit the loop (no need to iterate further repetitions)
            if exitflag_worker == 1:
                break

        # Communicate the results to the main process
        pipend.send((beta_worker, fval_beta_worker, exitflag_worker))

    def joint_optim(self, yi, q0, e0, beta0, n_rep):
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
        OUTPUTS:
            - beta_worker: ndarray
                optimized coefficients of the model. The shape is (2*self.n_parameters,).
        '''
        from scipy.optimize import minimize
        
        # First iteration
        res = minimize(
            lambda x: self.Jointloss(x, yi, q0, e0), beta0,
            method='SLSQP', options={'disp':False})
        beta_worker, exitflag_worker = res.x, int(res.success)
        
        # Iterate until the optimization is successful or the maximum number of repetitions is reached
        for _ in range(n_rep):
            res = minimize(
                lambda x: self.Jointloss(x, yi, q0, e0), beta_worker,
                method='SLSQP', options={'disp':False})
            beta_worker, exitflag_worker = res.x, int(res.success)
            #If optimization is successful, exit the loop (no need to iterate further repetitions)
            if exitflag_worker == 1:
                break

        return beta_worker

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
            res_train = self.fit(yi, seed=seed, return_train=True, q0=q0, nV=nV, n_init=n_init, n_rep=n_rep) #Train AE
            res_test = self.predict(yf)
            return {'qi':res_train['qi'], 'ei':res_train['ei'],
                    'qf':res_test['qf'], 'ef':res_test['ef'],
                    'beta':self.beta.reshape((2, self.n_parameters))} #Return prediction
        else:
            self.fit(yi, seed=seed, return_train=False, q0=q0, nV=nV, n_init=n_init, n_rep=n_rep) #Train AE
            res_test = self.predict(yf)
            return {'qf':res_test['qf'], 'ef':res_test['ef'],
                    'beta':self.beta.reshape((2, self.n_parameters))} #Return prediction

class CAESar_general(CAESar_base):
    '''
    CAESar for joint quantile and expected shortfall estimation
    '''
    def __init__(self, theta, spec='AS', lambdas=dict(), p=1, u=1):
        '''
        Initialization of the CAESar model.
        INPUTS:
            - theta: float
                desired confidence level.
            - spec: str, optional
                specification of the model (SAV, AS, GARCH). Default is AS.
            - lambdas: dict, optional
                lambdas for the soft constraints. Default is {'r':10, 'q':10, 'e':10}.
            - p: int, optional
                number of y_t lags for the model. Default is 1.
            - u: int, optional
                number of ^q_t lags for the model. Default is 1.
        OUTPUTS:
            - None.
        '''
        super().__init__(theta, lambdas) #Initialize the base class

        # Initialize p, u, and v
        self.p, self.u, self.max_lag = p, u, np.max([p, u])

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 1+p+2*u
            self.loop = self.R_SAVloop
            self.joint_loop = self.Joint_SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 1+2*p+2*u
            self.loop = self.R_ASloop
            self.joint_loop = self.Joint_ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 1+p+2*u
            self.loop = self.R_GARCHloop
            self.joint_loop = self.Joint_GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')

    def R_SAVloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via SAV specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (4,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and r at step -1 and we need to compute r at step 0
            r = list()
            r.append( beta[0] + beta[1] * np.abs(r0[0]) + beta[2] * r0[1] + beta[3] * r0[2] ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of r at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * r[t-1] )
        r = np.array(r)
        return r
    
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
            q.append( beta[0] + beta[1] * np.abs(e0[0]) + beta[2] * e0[1] + beta[3] * e0[2] ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( beta[4] + beta[5] * np.abs(e0[0]) + beta[6] * e0[1] + beta[7] * e0[2] )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * e[t-1] )
            e.append( beta[4] + beta[5] * np.abs(y[t-1]) + beta[6] * q[t-1] + beta[7] * e[t-1] )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def R_ASloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via AS specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (5,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and r at step -1 and we need to compute r at step 0
            # In pred_mode, r0 is assumed to be [y[-max_lag:], ^q[-max_lag:], ^r[-max_lag:]]
            r = list()
            for t in range(self.max_lag):
                r.append( r0[2][t])
            y = np.concatenate([r0[0], y])
            q = np.concatenate([r0[1], q])
            # Loop
            y_coeff_list = [np.where(y>0, beta[2*i-1], beta[2*i]) for i in range(1, self.p+1)]
            if len(y) > 0:
                for t in range(self.max_lag, len(y)):
                    r.append(beta[0] +\
                            np.sum([y_coeff_list[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                            np.sum([beta[self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) +\
                            np.sum([beta[self.p*2+self.u+j]*r[t-j] for j in range(1, self.u+1)]) )
            else:
                t = self.max_lag + 1
                r.append(beta[0] +\
                        np.sum([y_coeff_list[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                        np.sum([beta[self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) +\
                        np.sum([beta[self.p*2+self.u+j]*r[t-j] for j in range(1, self.u+1)]) )
            r = r[self.max_lag:]

        else: #If we are in training mode, we have an approximation of q at step 0
            r = [r0]*self.max_lag
            # Loop
            y_coeff_list = [np.where(y>0, beta[2*i-1], beta[2*i]) for i in range(1, self.p+1)]
            for t in range(self.max_lag, len(y)):
                r.append(beta[0] +\
                        np.sum([y_coeff_list[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                        np.sum([beta[self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) +\
                        np.sum([beta[self.p*2+self.u+j]*r[t-j] for j in range(1, self.u+1)]) )
        r = np.array(r)
        return r
    
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
            # In pred_mode, e0 is assumed to be [y[-max_lag:], ^q[-max_lag:], ^e[-max_lag:]]
            q = list()
            for t in range(self.max_lag):
                q.append( e0[1][t])
            e = list()
            for t in range(self.max_lag):
                e.append( e0[2][t])
            y = np.concatenate([e0[0], y])
            # Loop
            y_coeff_list_q = [np.where(y>0, beta[2*i-1], beta[2*i]) for i in range(1, self.p+1)]
            y_coeff_list_e = [np.where(y>0, beta[self.n_parameters+2*i-1],
                                       beta[self.n_parameters+2*i]) for i in range(1, self.p+1)]
            if len(y) > 0:
                for t in range(self.max_lag, len(y)):
                    q.append(beta[0] +\
                            np.sum([y_coeff_list_q[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                            np.sum([beta[self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) +\
                            np.sum([beta[self.p*2+self.u+j]*e[t-j] for j in range(1, self.u+1)]) )
                    e.append(beta[self.n_parameters] +\
                            np.sum([y_coeff_list_e[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                            np.sum([beta[self.n_parameters+self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) +\
                            np.sum([beta[self.n_parameters+self.p*2+self.u+j]*e[t-j] for j in range(1, self.u+1)]) )
            else:
                t = self.max_lag + 1
                q.append(beta[0] +\
                        np.sum([y_coeff_list_q[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                        np.sum([beta[self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) +\
                        np.sum([beta[self.p*2+self.u+j]*e[t-j] for j in range(1, self.u+1)]) )
                e.append(beta[self.n_parameters] +\
                        np.sum([y_coeff_list_e[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                        np.sum([beta[self.n_parameters+self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) +\
                        np.sum([beta[self.n_parameters+self.p*2+self.u+j]*e[t-j] for j in range(1, self.u+1)]) )
            q, e = q[self.max_lag:], e[self.max_lag:]

        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]*self.max_lag
            e = [e0]*self.max_lag
            # Loop
            y_coeff_list_q = [np.where(y>0, beta[2*i-1], beta[2*i]) for i in range(1, self.p+1)]
            y_coeff_list_e = [np.where(y>0, beta[self.n_parameters+2*i-1],
                                       beta[self.n_parameters+2*i]) for i in range(1, self.p+1)]
            for t in range(self.max_lag, len(y)):
                q.append(beta[0] +\
                        np.sum([y_coeff_list_q[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                        np.sum([beta[self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) +\
                        np.sum([beta[self.p*2+self.u+j]*e[t-j] for j in range(1, self.u+1)]) )
                e.append(beta[self.n_parameters] +\
                        np.sum([y_coeff_list_e[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                        np.sum([beta[self.n_parameters+self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) +\
                        np.sum([beta[self.n_parameters+self.p*2+self.u+j]*e[t-j] for j in range(1, self.u+1)]) )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def R_GARCHloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via GARCH specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (4,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute e at step 0
            r = list()
            r.append( -np.sqrt(beta[0]+beta[1]*r0[0]**2+beta[2]*r0[1]**2+beta[3]*r0[2]**2) ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of e at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*r[t-1]**2) )
        r = np.array(r)
        return r
    
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
            q.append( -np.sqrt(beta[0]+beta[1]*e0[0]**2+beta[2]*e0[1]**2+beta[3]*e0[2]**2) ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( -np.sqrt(beta[4]+beta[5]*e0[0]**2+beta[6]*e0[1]**2+beta[7]*e0[2]**2) )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*e[t-1]**2) )
            e.append( -np.sqrt(beta[4]+beta[5]*y[t-1]**2+beta[6]*q[t-1]**2+beta[7]*e[t-1]**2) )
        q, e = np.array(q), np.array(e)
        return q, e

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
        warnings.simplefilter(action='ignore', category=RuntimeWarning) #Ignore warnings

        #-------------------- Step 0: CAViaR for quantile initial guess
        cav_res = CAViaR(
            self.theta, self.mdl_spec, p=self.p, u=self.u).fit(
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
        AEfval = np.empty(nV) #Initialize the loss function vector
        #Iterates over the random initializations
        for i in range(nV):
            AEfval[i] = self.ESloss(beta0[i, :], yi, qi, e0-q0) #Compute starting loss
        beta0 = beta0[AEfval.argsort()][0:n_init] #Sort initializations by loss and select only the n_init best initializations
        
        #-------------------- Step 3: Optimization Routine - I step (barrera)
        beta = np.empty((n_init, self.n_parameters)) #Initialize the parameters vector
        fval_beta = np.empty(n_init) #Initialize the loss function vector
        exitflag = np.empty(n_init) #Initialize the exit flag vector

        # Multiprocessing: create and start worker processes
        workers = list()
        for i in range(n_init):
            parent_pipend, child_pipend = mp.Pipe() # Create a pipe to communicate with the worker
            worker = mp.Process(target=self.optim4mp,
                                args=(yi, qi, e0-q0, beta0[i, :],
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
        
        #-------------------- Step 4: Optimization Routine - II step (patton)
        # Concatenate ^q and ^r parameters, then convert ^r into ^e parameters
        joint_beta = np.concatenate([beta_cav, [0]*self.u, self.beta]) #Concatenate
        if self.mdl_spec == 'SAV':
            joint_beta[4] += joint_beta[0]
            joint_beta[5] += joint_beta[1]
            joint_beta[6] += joint_beta[2] - joint_beta[7] 
        elif self.mdl_spec == 'AS':
            for i in range(2*self.p +1):
                joint_beta[self.n_parameters + i] += joint_beta[i]
            for j in range(2*self.p +1, 2*self.p + self.u +1):
                joint_beta[self.n_parameters + j] += joint_beta[j] -\
                    joint_beta[self.n_parameters + self.u +j]
        elif self.mdl_spec == 'GARCH':
            joint_beta[4] += joint_beta[0]
            joint_beta[5] += joint_beta[1]
            joint_beta[6] += joint_beta[2] + joint_beta[7] 
            
        joint_beta_temp = self.joint_optim(yi, q0, e0, joint_beta, n_rep)
        if not np.isnan(joint_beta_temp).any():
            joint_beta = joint_beta_temp
        self.beta = joint_beta

        #Compute the fit output: optimal beta vector, optimization info, and the last pair ^q, ^e
        #in sample variables
        qi, ei = self.joint_loop(self.beta, yi, q0, e0) #Train CAESar and recover fitted quantiles
        self.last_state = [yi[-self.max_lag:], qi[-self.max_lag:], ei[-self.max_lag:]] #Store the last state

        if return_train: #If return_train is True, return the training prediction and coefficients
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
            self.last_state = [
                np.concatenate([self.last_state[0], yf])[-self.max_lag:],
                np.concatenate([self.last_state[1], qf])[-self.max_lag:],
                np.concatenate([self.last_state[2], ef])[-self.max_lag:] ]
        return {'qf':qf, 'ef':ef}

class CAESar_1_1(CAESar_base):
    '''
    CAESar for joint quantile and expected shortfall estimation - y lags=1, q lags=1.
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
                lambdas for the soft constraints. Default is {'r':10, 'q':10, 'e':10}.
        OUTPUTS:
            - None.
        '''
        super().__init__(theta, lambdas) #Initialize the base class

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

    def R_SAVloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via SAV specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (4,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and r at step -1 and we need to compute r at step 0
            r = list()
            r.append( beta[0] + beta[1] * np.abs(r0[0]) + beta[2] * r0[1] + beta[3] * r0[2] ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of r at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * r[t-1] )
        r = np.array(r)
        return r
    
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
            q.append( beta[0] + beta[1] * np.abs(e0[0]) + beta[2] * e0[1] + beta[3] * e0[2] ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( beta[4] + beta[5] * np.abs(e0[0]) + beta[6] * e0[1] + beta[7] * e0[2] )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * e[t-1] )
            e.append( beta[4] + beta[5] * np.abs(y[t-1]) + beta[6] * q[t-1] + beta[7] * e[t-1] )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def R_ASloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via AS specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (5,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
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
        r = np.array(r)
        return r
    
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
        q, e = np.array(q), np.array(e)
        return q, e
    
    def R_GARCHloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via GARCH specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (4,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute e at step 0
            r = list()
            r.append( -np.sqrt(beta[0]+beta[1]*r0[0]**2+beta[2]*r0[1]**2+beta[3]*r0[2]**2) ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of e at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*r[t-1]**2) )
        r = np.array(r)
        return r
    
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
            q.append( -np.sqrt(beta[0]+beta[1]*e0[0]**2+beta[2]*e0[1]**2+beta[3]*e0[2]**2) ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( -np.sqrt(beta[4]+beta[5]*e0[0]**2+beta[6]*e0[1]**2+beta[7]*e0[2]**2) )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*e[t-1]**2) )
            e.append( -np.sqrt(beta[4]+beta[5]*y[t-1]**2+beta[6]*q[t-1]**2+beta[7]*e[t-1]**2) )
        q, e = np.array(q), np.array(e)
        return q, e

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
        warnings.simplefilter(action='ignore', category=RuntimeWarning) #Ignore Runtime Warnings

        #-------------------- Step 0: CAViaR for quantile initial guess
        cav_res = CAViaR(
            self.theta, self.mdl_spec, p=1, u=1).fit(
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
        AEfval = np.empty(nV) #Initialize the loss function vector
        #Iterates over the random initializations
        for i in range(nV):
            AEfval[i] = self.ESloss(beta0[i, :], yi, qi, e0-q0) #Compute starting loss
        beta0 = beta0[AEfval.argsort()][0:n_init] #Sort initializations by loss and select only the n_init best initializations
        
        #-------------------- Step 3: Optimization Routine - I step (barrera)
        beta = np.empty((n_init, self.n_parameters)) #Initialize the parameters vector
        fval_beta = np.empty(n_init) #Initialize the loss function vector
        exitflag = np.empty(n_init) #Initialize the exit flag vector

        # Multiprocessing: create and start worker processes
        workers = list()
        for i in range(n_init):
            parent_pipend, child_pipend = mp.Pipe() # Create a pipe to communicate with the worker
            worker = mp.Process(target=self.optim4mp,
                                args=(yi, qi, e0-q0, beta0[i, :],
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
        
        #-------------------- Step 4: Optimization Routine - II step (patton)
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
            
        joint_beta_temp = self.joint_optim(yi, q0, e0, joint_beta, n_rep)
        if not np.isnan(joint_beta_temp).any():
            joint_beta = joint_beta_temp
        self.beta = joint_beta

        #Compute the fit output: optimal beta vector, optimization info, and the last pair ^q, ^e
        #in sample variables
        qi, ei = self.joint_loop(self.beta, yi, q0, e0) #Train CAESar and recover fitted quantiles
        self.last_state = [yi[-1], qi[-1], ei[-1]] #Store the last state

        if return_train: #If return_train is True, return the training prediction and coefficients
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

class CAESar_2_2(CAESar_base):
    '''
    CAESar for joint quantile and expected shortfall estimation - y lags=2, q lags=2.
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
                lambdas for the soft constraints. Default is {'r':10, 'q':10, 'e':10}.
        OUTPUTS:
            - None.
        '''
        super().__init__(theta, lambdas) #Initialize the base class

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 7
            self.loop = self.R_SAVloop
            self.joint_loop = self.Joint_SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 9
            self.loop = self.R_ASloop
            self.joint_loop = self.Joint_ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 7
            self.loop = self.R_GARCHloop
            self.joint_loop = self.Joint_GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')

    def R_SAVloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via SAV specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (4,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and r at step -1 and we need to compute r at step 0
            r = list()
            r.append( beta[0] + beta[1] * np.abs(r0[0]) + beta[2] * r0[1] + beta[3] * r0[2] ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of r at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * r[t-1] )
        r = np.array(r)
        return r
    
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
            q.append( beta[0] + beta[1] * np.abs(e0[0]) + beta[2] * e0[1] + beta[3] * e0[2] ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( beta[4] + beta[5] * np.abs(e0[0]) + beta[6] * e0[1] + beta[7] * e0[2] )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * e[t-1] )
            e.append( beta[4] + beta[5] * np.abs(y[t-1]) + beta[6] * q[t-1] + beta[7] * e[t-1] )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def R_ASloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via AS specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (5,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and r at step -1 and we need to compute r at step 0
            # In pred_mode, r0 is assumed to be [y[-max_lag:], ^q[-max_lag:], ^r[-max_lag:]]
            r = list()
            for t in range(2):
                r.append( r0[2][t])
            y = np.concatenate([r0[0], y])
            q = np.concatenate([r0[1], q])
            # Loop
            y_coeff_list_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_list_l2 = np.where(y>0, beta[3], beta[4])
            if len(y) > 0:
                for t in range(2, len(y)):
                    r.append(beta[0] +\
                            y_coeff_list_l1[t-1]*y[t-1] + y_coeff_list_l2[t-1]*y[t-2] +\
                            beta[5]*q[t-1] + beta[6]*q[t-2] +\
                            beta[7]*r[t-1] + beta[8]*r[t-2] )
            else:
                t = 3
                r.append(beta[0] +\
                        y_coeff_list_l1[t-1]*y[t-1] + y_coeff_list_l2[t-1]*y[t-2] +\
                        beta[5]*q[t-1] + beta[6]*q[t-2] +\
                        beta[7]*r[t-1] + beta[8]*r[t-2] )
            r = r[2:]

        else: #If we are in training mode, we have an approximation of q at step 0
            r = [r0]*2
            # Loop
            y_coeff_list_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_list_l2 = np.where(y>0, beta[3], beta[4])
            for t in range(2, len(y)):
                r.append(beta[0] +\
                        y_coeff_list_l1[t-1]*y[t-1] + y_coeff_list_l2[t-1]*y[t-2] +\
                        beta[5]*q[t-1] + beta[6]*q[t-2] +\
                        beta[7]*r[t-1] + beta[8]*r[t-2] )
        r = np.array(r)
        return r
    
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
            # In pred_mode, e0 is assumed to be [y[-max_lag:], ^q[-max_lag:], ^e[-max_lag:]]
            q = list()
            for t in range(2):
                q.append( e0[1][t])
            e = list()
            for t in range(2):
                e.append( e0[2][t])
            y = np.concatenate([e0[0], y])
            # Loop
            y_coeff_list_q_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_list_q_l2 = np.where(y>0, beta[3], beta[4])
            y_coeff_list_e_l1 = np.where(y>0, beta[10], beta[11])
            y_coeff_list_e_l2 = np.where(y>0, beta[12], beta[13])
            if len(y) > 0:
                for t in range(2, len(y)):
                    q.append(beta[0] +\
                            y_coeff_list_q_l1[t-1]*y[t-1] + y_coeff_list_q_l2[t-2]*y[t-2] +\
                            beta[5]*q[t-1] + beta[6]*q[t-2] +\
                            beta[7]*e[t-1] + beta[8]*e[t-2] )
                    e.append(beta[9] +\
                            y_coeff_list_e_l1[t-1]*y[t-1] + y_coeff_list_e_l2[t-2]*y[t-2] +\
                            beta[14]*q[t-1] + beta[15]*q[t-2] +\
                            beta[16]*e[t-1] + beta[17]*e[t-2] )
            else:
                t = 3
                q.append(beta[0] +\
                        y_coeff_list_q_l1[t-1]*y[t-1] + y_coeff_list_q_l2[t-2]*y[t-2] +\
                        beta[5]*q[t-1] + beta[6]*q[t-2] +\
                        beta[7]*e[t-1] + beta[8]*e[t-2] )
                e.append(beta[9] +\
                        y_coeff_list_e_l1[t-1]*y[t-1] + y_coeff_list_e_l2[t-2]*y[t-2] +\
                        beta[14]*q[t-1] + beta[15]*q[t-2] +\
                        beta[16]*e[t-1] + beta[17]*e[t-2] )
            q, e = q[2:], e[2:]

        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]*2
            e = [e0]*2
            # Loop
            y_coeff_list_q_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_list_q_l2 = np.where(y>0, beta[3], beta[4])
            y_coeff_list_e_l1 = np.where(y>0, beta[10], beta[11])
            y_coeff_list_e_l2 = np.where(y>0, beta[12], beta[13])
            for t in range(2, len(y)):
                q.append(beta[0] +\
                        y_coeff_list_q_l1[t-1]*y[t-1] + y_coeff_list_q_l2[t-2]*y[t-2] +\
                        beta[5]*q[t-1] + beta[6]*q[t-2] +\
                        beta[7]*e[t-1] + beta[8]*e[t-2] )
                e.append(beta[9] +\
                        y_coeff_list_e_l1[t-1]*y[t-1] + y_coeff_list_e_l2[t-2]*y[t-2] +\
                        beta[14]*q[t-1] + beta[15]*q[t-2] +\
                        beta[16]*e[t-1] + beta[17]*e[t-2] )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def R_GARCHloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via GARCH specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (4,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute e at step 0
            r = list()
            r.append( -np.sqrt(beta[0]+beta[1]*r0[0]**2+beta[2]*r0[1]**2+beta[3]*r0[2]**2) ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of e at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*r[t-1]**2) )
        r = np.array(r)
        return r
    
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
            q.append( -np.sqrt(beta[0]+beta[1]*e0[0]**2+beta[2]*e0[1]**2+beta[3]*e0[2]**2) ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( -np.sqrt(beta[4]+beta[5]*e0[0]**2+beta[6]*e0[1]**2+beta[7]*e0[2]**2) )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*e[t-1]**2) )
            e.append( -np.sqrt(beta[4]+beta[5]*y[t-1]**2+beta[6]*q[t-1]**2+beta[7]*e[t-1]**2) )
        q, e = np.array(q), np.array(e)
        return q, e

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
        warnings.simplefilter(action='ignore', category=RuntimeWarning) #Ignore Runtime Warnings

        #-------------------- Step 0: CAViaR for quantile initial guess
        cav_res = CAViaR(
            self.theta, self.mdl_spec, p=2, u=2).fit(
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
        AEfval = np.empty(nV) #Initialize the loss function vector
        #Iterates over the random initializations
        for i in range(nV):
            AEfval[i] = self.ESloss(beta0[i, :], yi, qi, e0-q0) #Compute starting loss
        beta0 = beta0[AEfval.argsort()][0:n_init] #Sort initializations by loss and select only the n_init best initializations
        
        #-------------------- Step 3: Optimization Routine - I step (barrera)
        beta = np.empty((n_init, self.n_parameters)) #Initialize the parameters vector
        fval_beta = np.empty(n_init) #Initialize the loss function vector
        exitflag = np.empty(n_init) #Initialize the exit flag vector

        # Multiprocessing: create and start worker processes
        workers = list()
        for i in range(n_init):
            parent_pipend, child_pipend = mp.Pipe() # Create a pipe to communicate with the worker
            worker = mp.Process(target=self.optim4mp,
                                args=(yi, qi, e0-q0, beta0[i, :],
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
        
        #-------------------- Step 4: Optimization Routine - II step (patton)
        # Concatenate ^q and ^r parameters, then convert ^r into ^e parameters
        joint_beta = np.concatenate([beta_cav, [0, 0], self.beta]) #Concatenate
        if self.mdl_spec == 'SAV':
            joint_beta[4] += joint_beta[0]
            joint_beta[5] += joint_beta[1]
            joint_beta[6] += joint_beta[2] - joint_beta[7] 
        elif self.mdl_spec == 'AS':
            for i in range(5):
                joint_beta[9 + i] += joint_beta[i]
            for j in range(5, 7):
                joint_beta[9 + j] += joint_beta[j] - joint_beta[11 +j]
        elif self.mdl_spec == 'GARCH':
            joint_beta[4] += joint_beta[0]
            joint_beta[5] += joint_beta[1]
            joint_beta[6] += joint_beta[2] + joint_beta[7] 
            
        joint_beta_temp = self.joint_optim(yi, q0, e0, joint_beta, n_rep)
        if not np.isnan(joint_beta_temp).any():
            joint_beta = joint_beta_temp
        self.beta = joint_beta

        #Compute the fit output: optimal beta vector, optimization info, and the last pair ^q, ^e
        #in sample variables
        qi, ei = self.joint_loop(self.beta, yi, q0, e0) #Train CAESar and recover fitted quantiles
        self.last_state = [yi[-2:], qi[-2:], ei[-2:]] #Store the last state

        if return_train: #If return_train is True, return the training prediction and coefficients
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
            self.last_state = [
                np.concatenate([self.last_state[0], yf])[-2:],
                np.concatenate([self.last_state[1], qf])[-2:],
                np.concatenate([self.last_state[2], ef])[-2:] ]
        return {'qf':qf, 'ef':ef}

class CAESar_1_2(CAESar_base):
    '''
    CAESar for joint quantile and expected shortfall estimation - y lags=1, q lags=2.
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
                lambdas for the soft constraints. Default is {'r':10, 'q':10, 'e':10}.
        OUTPUTS:
            - None.
        '''
        super().__init__(theta, lambdas) #Initialize the base class

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 6
            self.loop = self.R_SAVloop
            self.joint_loop = self.Joint_SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 7
            self.loop = self.R_ASloop
            self.joint_loop = self.Joint_ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 6
            self.loop = self.R_GARCHloop
            self.joint_loop = self.Joint_GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')

    def R_SAVloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via SAV specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (4,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and r at step -1 and we need to compute r at step 0
            r = list()
            r.append( beta[0] + beta[1] * np.abs(r0[0]) + beta[2] * r0[1] + beta[3] * r0[2] ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of r at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * r[t-1] )
        r = np.array(r)
        return r
    
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
            q.append( beta[0] + beta[1] * np.abs(e0[0]) + beta[2] * e0[1] + beta[3] * e0[2] ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( beta[4] + beta[5] * np.abs(e0[0]) + beta[6] * e0[1] + beta[7] * e0[2] )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * e[t-1] )
            e.append( beta[4] + beta[5] * np.abs(y[t-1]) + beta[6] * q[t-1] + beta[7] * e[t-1] )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def R_ASloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via AS specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (5,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and r at step -1 and we need to compute r at step 0
            # In pred_mode, r0 is assumed to be [y[-max_lag:], ^q[-max_lag:], ^r[-max_lag:]]
            r = list()
            for t in range(2):
                r.append( r0[2][t])
            y = np.concatenate([r0[0], y])
            q = np.concatenate([r0[1], q])
            # Loop
            y_coeff_list_l1 = np.where(y>0, beta[1], beta[2])
            if len(y) > 0:
                for t in range(2, len(y)):
                    r.append(beta[0] +\
                            y_coeff_list_l1[t-1]*y[t-1] +\
                            beta[3]*q[t-1] + beta[4]*q[t-2] +\
                            beta[5]*r[t-1] + beta[6]*r[t-2] )
            else:
                t = 3
                r.append(beta[0] +\
                        y_coeff_list_l1[t-1]*y[t-1] +\
                        beta[3]*q[t-1] + beta[4]*q[t-2] +\
                        beta[5]*r[t-1] + beta[6]*r[t-2] )
            r = r[2:]

        else: #If we are in training mode, we have an approximation of q at step 0
            r = [r0]*2
            # Loop
            y_coeff_list_l1 = np.where(y>0, beta[1], beta[2])
            for t in range(2, len(y)):
                r.append(beta[0] +\
                        y_coeff_list_l1[t-1]*y[t-1] +\
                        beta[3]*q[t-1] + beta[4]*q[t-2] +\
                        beta[5]*r[t-1] + beta[6]*r[t-2] )
        r = np.array(r)
        return r
    
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
            # In pred_mode, e0 is assumed to be [y[-max_lag:], ^q[-max_lag:], ^e[-max_lag:]]
            q = list()
            for t in range(2):
                q.append( e0[1][t])
            e = list()
            for t in range(2):
                e.append( e0[2][t])
            y = np.concatenate([e0[0], y])
            # Loop
            y_coeff_list_q_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_list_e_l1 = np.where(y>0, beta[8], beta[9])
            if len(y) > 0:
                for t in range(2, len(y)):
                    q.append(beta[0] +\
                            y_coeff_list_q_l1[t-1]*y[t-1] +\
                            beta[3]*q[t-1] + beta[4]*q[t-2] +\
                            beta[5]*e[t-1] + beta[6]*e[t-2] )
                    e.append(beta[7] +\
                            y_coeff_list_e_l1[t-1]*y[t-1] +\
                            beta[10]*q[t-1] + beta[11]*q[t-2] +\
                            beta[12]*e[t-1] + beta[13]*e[t-2] )
            else:
                t = 3
                q.append(beta[0] +\
                        y_coeff_list_q_l1[t-1]*y[t-1] +\
                        beta[3]*q[t-1] + beta[4]*q[t-2] +\
                        beta[5]*e[t-1] + beta[6]*e[t-2] )
                e.append(beta[7] +\
                        y_coeff_list_e_l1[t-1]*y[t-1] +\
                        beta[10]*q[t-1] + beta[11]*q[t-2] +\
                        beta[12]*e[t-1] + beta[13]*e[t-2] )
            q, e = q[2:], e[2:]

        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]*2
            e = [e0]*2
            # Loop
            y_coeff_list_q_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_list_e_l1 = np.where(y>0, beta[8], beta[9])
            for t in range(2, len(y)):
                q.append(beta[0] +\
                        y_coeff_list_q_l1[t-1]*y[t-1] +\
                        beta[3]*q[t-1] + beta[4]*q[t-2] +\
                        beta[5]*e[t-1] + beta[6]*e[t-2] )
                e.append(beta[7] +\
                        y_coeff_list_e_l1[t-1]*y[t-1] +\
                        beta[10]*q[t-1] + beta[11]*q[t-2] +\
                        beta[12]*e[t-1] + beta[13]*e[t-2] )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def R_GARCHloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via GARCH specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (4,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute e at step 0
            r = list()
            r.append( -np.sqrt(beta[0]+beta[1]*r0[0]**2+beta[2]*r0[1]**2+beta[3]*r0[2]**2) ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of e at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*r[t-1]**2) )
        r = np.array(r)
        return r
    
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
            q.append( -np.sqrt(beta[0]+beta[1]*e0[0]**2+beta[2]*e0[1]**2+beta[3]*e0[2]**2) ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( -np.sqrt(beta[4]+beta[5]*e0[0]**2+beta[6]*e0[1]**2+beta[7]*e0[2]**2) )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*e[t-1]**2) )
            e.append( -np.sqrt(beta[4]+beta[5]*y[t-1]**2+beta[6]*q[t-1]**2+beta[7]*e[t-1]**2) )
        q, e = np.array(q), np.array(e)
        return q, e

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
        warnings.simplefilter(action='ignore', category=RuntimeWarning) #Ignore Runtime Warnings

        #-------------------- Step 0: CAViaR for quantile initial guess
        cav_res = CAViaR(
            self.theta, self.mdl_spec, p=1, u=2).fit(
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
        AEfval = np.empty(nV) #Initialize the loss function vector
        #Iterates over the random initializations
        for i in range(nV):
            AEfval[i] = self.ESloss(beta0[i, :], yi, qi, e0-q0) #Compute starting loss
        beta0 = beta0[AEfval.argsort()][0:n_init] #Sort initializations by loss and select only the n_init best initializations
        
        #-------------------- Step 3: Optimization Routine - I step (barrera)
        beta = np.empty((n_init, self.n_parameters)) #Initialize the parameters vector
        fval_beta = np.empty(n_init) #Initialize the loss function vector
        exitflag = np.empty(n_init) #Initialize the exit flag vector

        # Multiprocessing: create and start worker processes
        workers = list()
        for i in range(n_init):
            parent_pipend, child_pipend = mp.Pipe() # Create a pipe to communicate with the worker
            worker = mp.Process(target=self.optim4mp,
                                args=(yi, qi, e0-q0, beta0[i, :],
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
        
        #-------------------- Step 4: Optimization Routine - II step (patton)
        # Concatenate ^q and ^r parameters, then convert ^r into ^e parameters
        joint_beta = np.concatenate([beta_cav, [0, 0], self.beta]) #Concatenate
        if self.mdl_spec == 'SAV':
            joint_beta[4] += joint_beta[0]
            joint_beta[5] += joint_beta[1]
            joint_beta[6] += joint_beta[2] - joint_beta[7] 
        elif self.mdl_spec == 'AS':
            for i in range(3):
                joint_beta[7 + i] += joint_beta[i]
            for j in range(3, 5):
                joint_beta[7 + j] += joint_beta[j] - joint_beta[9 +j]
        elif self.mdl_spec == 'GARCH':
            joint_beta[4] += joint_beta[0]
            joint_beta[5] += joint_beta[1]
            joint_beta[6] += joint_beta[2] + joint_beta[7] 
            
        joint_beta_temp = self.joint_optim(yi, q0, e0, joint_beta, n_rep)
        if not np.isnan(joint_beta_temp).any():
            joint_beta = joint_beta_temp
        self.beta = joint_beta

        #Compute the fit output: optimal beta vector, optimization info, and the last pair ^q, ^e
        #in sample variables
        qi, ei = self.joint_loop(self.beta, yi, q0, e0) #Train CAESar and recover fitted quantiles
        self.last_state = [yi[-2:], qi[-2:], ei[-2:]] #Store the last state

        if return_train: #If return_train is True, return the training prediction and coefficients
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
            self.last_state = [
                np.concatenate([self.last_state[0], yf])[-2:],
                np.concatenate([self.last_state[1], qf])[-2:],
                np.concatenate([self.last_state[2], ef])[-2:] ]
        return {'qf':qf, 'ef':ef}

class CAESar_2_1(CAESar_base):
    '''
    CAESar for joint quantile and expected shortfall estimation - y lags=2, q lags=1.
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
                lambdas for the soft constraints. Default is {'r':10, 'q':10, 'e':10}.
        OUTPUTS:
            - None.
        '''
        super().__init__(theta, lambdas) #Initialize the base class

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 5
            self.loop = self.R_SAVloop
            self.joint_loop = self.Joint_SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 7
            self.loop = self.R_ASloop
            self.joint_loop = self.Joint_ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 5
            self.loop = self.R_GARCHloop
            self.joint_loop = self.Joint_GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')

    def R_SAVloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via SAV specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (4,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and r at step -1 and we need to compute r at step 0
            r = list()
            r.append( beta[0] + beta[1] * np.abs(r0[0]) + beta[2] * r0[1] + beta[3] * r0[2] ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of r at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * r[t-1] )
        r = np.array(r)
        return r
    
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
            q.append( beta[0] + beta[1] * np.abs(e0[0]) + beta[2] * e0[1] + beta[3] * e0[2] ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( beta[4] + beta[5] * np.abs(e0[0]) + beta[6] * e0[1] + beta[7] * e0[2] )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1] * np.abs(y[t-1]) + beta[2] * q[t-1] + beta[3] * e[t-1] )
            e.append( beta[4] + beta[5] * np.abs(y[t-1]) + beta[6] * q[t-1] + beta[7] * e[t-1] )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def R_ASloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via AS specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (5,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and r at step -1 and we need to compute r at step 0
            # In pred_mode, r0 is assumed to be [y[-max_lag:], ^q[-max_lag:], ^r[-max_lag:]]
            r = list()
            for t in range(2):
                r.append( r0[2][t])
            y = np.concatenate([r0[0], y])
            q = np.concatenate([r0[1], q])
            # Loop
            y_coeff_list_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_list_l2 = np.where(y>0, beta[3], beta[4])
            if len(y) > 0:
                for t in range(2, len(y)):
                    r.append(beta[0] +\
                            y_coeff_list_l1[t-1]*y[t-1] + y_coeff_list_l2[t-1]*y[t-2] +\
                            beta[5]*q[t-1] +\
                            beta[6]*r[t-1] )
            else:
                t = 3
                r.append(beta[0] +\
                        y_coeff_list_l1[t-1]*y[t-1] + y_coeff_list_l2[t-1]*y[t-2] +\
                        beta[5]*q[t-1] +\
                        beta[6]*r[t-1] )
            r = r[2:]

        else: #If we are in training mode, we have an approximation of q at step 0
            r = [r0]*2
            # Loop
            y_coeff_list_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_list_l2 = np.where(y>0, beta[3], beta[4])
            for t in range(2, len(y)):
                r.append(beta[0] +\
                        y_coeff_list_l1[t-1]*y[t-1] + y_coeff_list_l2[t-1]*y[t-2] +\
                        beta[5]*q[t-1] +\
                        beta[6]*r[t-1] )
        r = np.array(r)
        return r
    
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
            # In pred_mode, e0 is assumed to be [y[-max_lag:], ^q[-max_lag:], ^e[-max_lag:]]
            q = list()
            for t in range(2):
                q.append( e0[1][t])
            e = list()
            for t in range(2):
                e.append( e0[2][t])
            y = np.concatenate([e0[0], y])
            # Loop
            y_coeff_list_q_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_list_q_l2 = np.where(y>0, beta[3], beta[4])
            y_coeff_list_e_l1 = np.where(y>0, beta[8], beta[9])
            y_coeff_list_e_l2 = np.where(y>0, beta[10], beta[11])
            if len(y) > 0:
                for t in range(2, len(y)):
                    q.append(beta[0] +\
                            y_coeff_list_q_l1[t-1]*y[t-1] + y_coeff_list_q_l2[t-2]*y[t-2] +\
                            beta[5]*q[t-1] +\
                            beta[6]*e[t-1] )
                    e.append(beta[7] +\
                            y_coeff_list_e_l1[t-1]*y[t-1] + y_coeff_list_e_l2[t-2]*y[t-2] +\
                            beta[12]*q[t-1] +\
                            beta[13]*e[t-1] )
            else:
                t = 3
                q.append(beta[0] +\
                        y_coeff_list_q_l1[t-1]*y[t-1] + y_coeff_list_q_l2[t-2]*y[t-2] +\
                        beta[5]*q[t-1] +\
                        beta[6]*e[t-1] )
                e.append(beta[7] +\
                        y_coeff_list_e_l1[t-1]*y[t-1] + y_coeff_list_e_l2[t-2]*y[t-2] +\
                        beta[12]*q[t-1] +\
                        beta[13]*e[t-1] )
            q, e = q[2:], e[2:]

        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]*2
            e = [e0]*2
            # Loop
            y_coeff_list_q_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_list_q_l2 = np.where(y>0, beta[3], beta[4])
            y_coeff_list_e_l1 = np.where(y>0, beta[8], beta[9])
            y_coeff_list_e_l2 = np.where(y>0, beta[10], beta[11])
            for t in range(2, len(y)):
                q.append(beta[0] +\
                        y_coeff_list_q_l1[t-1]*y[t-1] + y_coeff_list_q_l2[t-2]*y[t-2] +\
                        beta[5]*q[t-1] +\
                        beta[6]*e[t-1] )
                e.append(beta[7] +\
                        y_coeff_list_e_l1[t-1]*y[t-1] + y_coeff_list_e_l2[t-2]*y[t-2] +\
                        beta[12]*q[t-1] +\
                        beta[13]*e[t-1] )
        q, e = np.array(q), np.array(e)
        return q, e
    
    def R_GARCHloop(self, beta, y, q, r0, pred_mode=False):
        '''
        Loop for the ^r estimation via GARCH specification.
        INPUTS:
            - beta: ndarray
                parameters of the model. The shape is (4,).
            - y: ndarray
                target time series.
            - q: ndarray
                quantile forecast.
            - r0: float or list of floats
                if pred_mode=False, r0 is a float describing initial point for the
                ^r forecast. If pred_mode=True, r0 is a list of floats describing the
                last state of the system, that is r0=[y[-1], q[-1], ^r[-1]].
            - pred_mode: bool, optional
                if True, the loop is in prediction mode and r0 is assumed to contain
                the last state. Default is False.
        OUTPUTS:
            - r: ndarray
                difference between the Expected Shortfall forecast and the quantile.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y, q, and e at step -1 and we need to compute e at step 0
            r = list()
            r.append( -np.sqrt(beta[0]+beta[1]*r0[0]**2+beta[2]*r0[1]**2+beta[3]*r0[2]**2) ) #In pred_mode, r0 is assumed to be [y[-1], q[-1], r[-1]]
        else: #If we are in training mode, we have an approximation of e at step 0
            r = [r0]

        # Loop
        for t in range(1, len(y)):
            r.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*r[t-1]**2) )
        r = np.array(r)
        return r
    
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
            q.append( -np.sqrt(beta[0]+beta[1]*e0[0]**2+beta[2]*e0[1]**2+beta[3]*e0[2]**2) ) #In pred_mode, e0 is assumed to be [y[-1], q[-1], e[-1]]
            e.append( -np.sqrt(beta[4]+beta[5]*e0[0]**2+beta[6]*e0[1]**2+beta[7]*e0[2]**2) )
        else: #If we are in training mode, we have an approximation of q and e at step 0
            q = [q0]
            e = [e0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0]+beta[1]*y[t-1]**2+beta[2]*q[t-1]**2+beta[3]*e[t-1]**2) )
            e.append( -np.sqrt(beta[4]+beta[5]*y[t-1]**2+beta[6]*q[t-1]**2+beta[7]*e[t-1]**2) )
        q, e = np.array(q), np.array(e)
        return q, e

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
        warnings.simplefilter(action='ignore', category=RuntimeWarning) #Ignore Runtime Warnings

        #-------------------- Step 0: CAViaR for quantile initial guess
        cav_res = CAViaR(
            self.theta, self.mdl_spec, p=2, u=1).fit(
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
        AEfval = np.empty(nV) #Initialize the loss function vector
        #Iterates over the random initializations
        for i in range(nV):
            AEfval[i] = self.ESloss(beta0[i, :], yi, qi, e0-q0) #Compute starting loss
        beta0 = beta0[AEfval.argsort()][0:n_init] #Sort initializations by loss and select only the n_init best initializations
        
        #-------------------- Step 3: Optimization Routine - I step (barrera)
        beta = np.empty((n_init, self.n_parameters)) #Initialize the parameters vector
        fval_beta = np.empty(n_init) #Initialize the loss function vector
        exitflag = np.empty(n_init) #Initialize the exit flag vector

        # Multiprocessing: create and start worker processes
        workers = list()
        for i in range(n_init):
            parent_pipend, child_pipend = mp.Pipe() # Create a pipe to communicate with the worker
            worker = mp.Process(target=self.optim4mp,
                                args=(yi, qi, e0-q0, beta0[i, :],
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
        
        #-------------------- Step 4: Optimization Routine - II step (patton)
        # Concatenate ^q and ^r parameters, then convert ^r into ^e parameters
        joint_beta = np.concatenate([beta_cav, [0], self.beta]) #Concatenate
        if self.mdl_spec == 'SAV':
            joint_beta[4] += joint_beta[0]
            joint_beta[5] += joint_beta[1]
            joint_beta[6] += joint_beta[2] - joint_beta[7] 
        elif self.mdl_spec == 'AS':
            for i in range(5):
                joint_beta[7 + i] += joint_beta[i]
            for j in range(5, 6):
                joint_beta[7 + j] += joint_beta[j] - joint_beta[8 +j]
        elif self.mdl_spec == 'GARCH':
            joint_beta[4] += joint_beta[0]
            joint_beta[5] += joint_beta[1]
            joint_beta[6] += joint_beta[2] + joint_beta[7] 
            
        joint_beta_temp = self.joint_optim(yi, q0, e0, joint_beta, n_rep)
        if not np.isnan(joint_beta_temp).any():
            joint_beta = joint_beta_temp
        self.beta = joint_beta

        #Compute the fit output: optimal beta vector, optimization info, and the last pair ^q, ^e
        #in sample variables
        qi, ei = self.joint_loop(self.beta, yi, q0, e0) #Train CAESar and recover fitted quantiles
        self.last_state = [yi[-2:], qi[-2:], ei[-2:]] #Store the last state

        if return_train: #If return_train is True, return the training prediction and coefficients
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
            self.last_state = [
                np.concatenate([self.last_state[0], yf])[-2:],
                np.concatenate([self.last_state[1], qf])[-2:],
                np.concatenate([self.last_state[2], ef])[-2:] ]
        return {'qf':qf, 'ef':ef}

def CAESar(theta, spec='AS', lambdas=dict(), p=1, u=1):
    '''
    CAESar model selection.
    INPUTS:
        - theta: float
            quantile level.
        - spec: str, optional
            specification of the model (SAV, AS, GARCH). Default is AS.
        - p: int, optional
            number of lags for the y variable. Default is 1.
        - u: int, optional
            number of lags for ^q and ^e. Default is 1.
    OUTPUTS:
        - CAESar model
    '''
    p, u = int(p), int(u) #Convert to integer
    # Ensure that p and r are greater than 0
    assert p > 0, 'p must be greater than 0'
    assert u > 0, 'u must be greater than 0'
    # Check for optimized versions
    if p == 1:
        if u == 1:
            return CAESar_1_1(theta, spec=spec, lambdas=lambdas)
        elif u == 2:
            return CAESar_1_2(theta, spec=spec, lambdas=lambdas)
    elif p == 2:
        if u == 1:
            return CAESar_2_1(theta, spec=spec, lambdas=lambdas)
        elif u == 2:
            return CAESar_2_2(theta, spec=spec, lambdas=lambdas)
    # Otherwise, use thee general version and allert the user
    warnings.warn('The selected model is not optimized. Consider using the optimized versions for p and u in {1,2}.')
    return CAESar_general(theta, spec=spec, lambdas=lambdas, p=p, u=u)
