
import numpy as np
import warnings

class CAViaR_base():
    def __init__(self):
        pass
    
    def loss_function(self, q, y):
        '''
        Pinball loss function.
        INPUTS:
            - q: ndarray
                quantile estimate.
            - y: ndarray
                true value.
        OUTPUTS:
            - Pinball loss: float
                loss value.
        '''
        return np.mean(
            np.where(y<q, (1-self.theta), -self.theta) * (q-y)
        )

    def Qloss(self, beta, y, q0):
        '''
        Loss function for the CAViaR model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
        OUTPUTS:
            - loss: float
                loss value.
        '''
        q = self.loop(beta, y, q0)
        return self.loss_function(q, y)

    def fit(self, yi, seed=None, q0=None):
        '''
        Fit the CAViaR model.
        INPUTS:
            - yi: ndarray
                training data.
            - seed: int or None, optional
                random seed. Default is None.
            - q0: float, optional
                initial quantile. Default is None.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
        '''
        from scipy.optimize import fmin

        warnings.simplefilter(action='ignore', category=RuntimeWarning) #Ignore Runtime Warnings

        #-------------------- Step 1: Initialization
        if isinstance(yi, list):
            yi = np.array(yi)
        nV, nC, n_rep = 102, 2, 5 #Set the number of: initial vectors; initial parameters; repetitions
        
        # Check if the starting point ^q_0 is provided by the user or has to be estimated
        if isinstance(q0, type(None)):
            #The starting forecast ^q_0 is the empricial quantile of the first part of the trainig set (for computational reason)
            n_emp = int(np.ceil(0.1 * len(yi))) #Select onyl 1/10 of the training set
            if round(n_emp * self.theta) == 0: n_emp = len(yi) #In case the training dimension is too small wrt theta
            q0 = np.sort(yi[0:n_emp])[int(round(n_emp * self.theta))-1]

        #-------------------- Step 2: Initial guess
        np.random.seed(seed)
        nInitialVectors = [nV//3, self.n_parameters] #Define the shape of the random initializations
        beta0 = [np.random.uniform(0, 1, nInitialVectors)]
        beta0.append(np.random.uniform(-1, 1, nInitialVectors))
        beta0.append(np.random.randn(*nInitialVectors))
        beta0 = np.concatenate(beta0, axis=0)
        Qfval = np.empty(nV) #Initialize the loss function vector
        #Iterates over the random initializations
        for i in range(nV):
            Qfval[i] = self.Qloss(beta0[i, :], yi, q0) #Compute starting loss
        beta0 = beta0[Qfval.argsort()][0:nC] #Sort initializations by loss and select only the nC best initializations

        #-------------------- Step 3: Optimization Routine
        beta = np.empty((nC, self.n_parameters)) #Initialize the parameters vector
        fval_beta = np.empty(nC) #Initialize the loss function vector
        exitflag = np.empty(nC) #Initialize the exit flag vector
        for i in range(nC):
            # First iteration
            beta[i, :], fval_beta[i], _, _, temp = fmin(
                lambda x: self.Qloss(x, yi, q0), beta0[i, :],
                disp=False, full_output=True, ftol=1e-7)
            exitflag[i] = 1 if temp==0 else 0

            # Iterate until the optimization is successful or the maximum number of repetitions is reached
            for _ in range(n_rep):
                beta[i, :], fval_beta[i], _, _, temp = fmin(
                    lambda x: self.Qloss(x, yi, q0),
                    beta[i, :], disp=False, full_output=True, ftol=1e-7) #Minimize over beta
                exitflag[i] = 1 if temp==0 else 0
                #If optimization is successful, exit the loop (no need to iterate further repetitions)
                if exitflag[i] == 1:
                    break

        #Compute the fit output: optimal beta vector, optimization info, and the last ^q
        ind_min = np.argmin(fval_beta) #Select the index of the best loss
        self.beta = beta[ind_min, :] #Store the best parameters
        self.fval_beta = fval_beta[ind_min] #Store the best loss
        self.exitflag = exitflag[ind_min] #Store the best exit flag

        qi = self.loop(self.beta, yi, q0)
        return qi

    def fit_predict(self, y, ti, seed=None, return_train=True, q0=None):
        '''
        Fit the model and predict the quantile.
        INPUTS:
            - y: ndarray
                data.
            - ti: int
                train/test split point.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                return the training prediction. Default is True.
            - q0: float, optional
                initial quantile. Default is None.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - qf: ndarray
                quantile forecast in the test set
            - beta: ndarray
                fitted coefficients of the regression
        '''
        yi, yf = y[:ti], y[ti:] #Split train and test
        if return_train:
            res_train = self.fit(yi, seed=seed, return_train=True, q0=q0) #Train CAViaR
            qf = self.predict(yf) #Predict
            return {'qi':res_train['qi'], 'qf':qf, 'beta':self.beta} #Return prediction
        else:
            self.fit(yi, seed=seed, return_train=False, q0=q0) #Train AE
            qf = self.predict(yf)
            return {'qf':qf, 'beta':self.beta} #Return prediction

class CAViaR_general(CAViaR_base):
    '''
    CAViaR regression for quantile estimation.
    '''
    def __init__(self, theta, spec='AS', p=1, u=1):
        '''
        Initialization of the general CAViaR model.
        INPUTS:
            - theta: float
                quantile level.
            - spec: str, optional
                specification of the model (SAV, AS, GARCH). Default is AS.
            - p: int, optional
                number of y_t lags for the model. Default is 1.
            - u: int, optional
                number of ^q_t lags for the model. Default is 1.
        OUTPUTS:
            - None
        '''
        self.theta = theta #Initialize theta

        # Initialize p and u
        self.p, self.u, self.max_lag = p, u, np.max([p, u])

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 1+p+u
            self.loop = self.SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 1+2*p+u
            self.loop = self.ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 1+p+u
            self.loop = self.GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')
    
    def SAVloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the SAV model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            q = list()
            q.append( beta[0] + beta[1] * np.abs(q0[0]) + beta[2] * q0[1] ) #In pred_mode, q0 is assumed to be [y[-1], q[-1]]
        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1]*np.abs(y[t-1]) + beta[2]*q[t-1] )

        return np.array(q)

    def ASloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the AS model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            # In pred_mode, q0 is assumed to be [y[-max([p, r]):], ^q[-max([p, r]):]]
            q = list()
            for t in range(self.max_lag):
                q.append( q0[1][t])
            y = np.concatenate([q0[0], y])
            # Loop
            y_coeff_list = [np.where(y>0, beta[2*i-1], beta[2*i]) for i in range(1, self.p+1)]
            if len(y) > 0:
                for t in range(self.max_lag, len(y)):
                    q.append(beta[0] +\
                            np.sum([y_coeff_list[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                            np.sum([beta[self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) )
            else:
                t = self.max_lag + 1
                q.append(beta[0] +\
                        np.sum([y_coeff_list[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                        np.sum([beta[self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) )
            q = q[self.max_lag:]

        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]*self.max_lag
            # Loop
            y_coeff_list = [np.where(y>0, beta[2*i-1], beta[2*i]) for i in range(1, self.p+1)]
            for t in range(self.max_lag, len(y)):
                q.append(beta[0] +\
                        np.sum([y_coeff_list[i-1][t-i]*y[t-i] for i in range(1, self.p+1)]) +\
                        np.sum([beta[self.p*2+j]*q[t-j] for j in range(1, self.u+1)]) )
        q = np.array(q)
        return q

    def GARCHloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the GARCH model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            q = list()
            q.append( -np.sqrt(beta[0] + beta[1]*q0[0]**2 + beta[2]*q0[1]**2) ) #In pred_mode, q0 is assumed to be [y[-1], q[-1]]
        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0] + beta[1]*y[t-1]**2 + beta[2]*q[t-1]**2) )
        return q

    def fit(self, yi, seed=None, return_train=False, q0=None):
        '''
        Fit the CAViaR model.
        INPUTS:
            - yi: ndarray
                training data.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                return the training prediction. Default is False.
            - q0: float, optional
                initial quantile. Default is None.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - beta: ndarray
                fitted coefficients of the regression (if return_train=True).
        '''
        
        qi = super().fit(yi, seed=seed, q0=q0) #Train CAViaR and recover fitted quantiles
        self.last_state = [yi[-self.max_lag:], qi[-self.max_lag:]] #Store the last state
        
        if return_train: #If return_train is True, return the training prediction and coefficients
            return {'qi':qi, 'beta':self.beta}
    
    def predict(self, yf=np.array(list())):
        '''
        Predict the quantile.
        INPUTS:
            - yf: ndarray, optional
                test data. If yf is not empty, the internal state is updated
                with the last observation.Default is an empty list.
        OUTPUTS:
            - qf: ndarray
                quantile estimate.
        '''
        qf = self.loop(self.beta, yf, self.last_state, pred_mode=True)
        if len(yf) > 0:
            self.last_state = [
                np.concatenate([self.last_state[0], yf])[-self.max_lag:],
                np.concatenate([self.last_state[1], qf])[-self.max_lag:] ]
        return qf

class CAViaR_1_1(CAViaR_base):
    '''
    CAViaR regression for quantile estimation - y lags=1, q lags=1.
    '''
    def __init__(self, theta, spec='AS'):
        '''
        Initialization of the CAViaR model.
        INPUTS:
            - theta: float
                quantile level.
            - spec: str, optional
                specification of the model (SAV, AS, GARCH). Default is AS.
        OUTPUTS:
            - None
        '''
        self.theta = theta #Initialize theta

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 3
            self.loop = self.SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 4
            self.loop = self.ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 3
            self.loop = self.GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')
    
    def SAVloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the SAV model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            q = list()
            q.append( beta[0] + beta[1] * np.abs(q0[0]) + beta[2] * q0[1] ) #In pred_mode, q0 is assumed to be [y[-1], q[-1]]
        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1]*np.abs(y[t-1]) + beta[2]*q[t-1] )

        return np.array(q)

    def ASloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the AS model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            q = list()
            q.append( beta[0] + np.where(q0[0]>0, beta[1], beta[2]) * q0[0] + beta[3] * q0[1] ) #In pred_mode, q0 is assumed to be [y[-1], q[-1]]
        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]

        # Loop
        y_plus_coeff = np.where(y>0, beta[1], beta[2]) #Only one between positive and negative y part gives a contribution
        for t in range(1, len(y)):
            q.append( beta[0] + y_plus_coeff[t-1]*y[t-1] + beta[3]*q[t-1] )
        return np.array(q)

    def GARCHloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the GARCH model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            q = list()
            q.append( -np.sqrt(beta[0] + beta[1]*q0[0]**2 + beta[2]*q0[1]**2) ) #In pred_mode, q0 is assumed to be [y[-1], q[-1]]
        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0] + beta[1]*y[t-1]**2 + beta[2]*q[t-1]**2) )
        return q

    def fit(self, yi, seed=None, return_train=False, q0=None):
        '''
        Fit the CAViaR model.
        INPUTS:
            - yi: ndarray
                training data.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                return the training prediction. Default is False.
            - q0: float, optional
                initial quantile. Default is None.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - beta: ndarray
                fitted coefficients of the regression (if return_train=True).
        '''
        
        qi = super().fit(yi, seed=seed, q0=q0) #Train CAViaR and recover fitted quantiles
        self.last_state = [yi[-1], qi[-1]] #Store the last state

        if return_train: #If return_train is True, return the training prediction and coefficients
            return {'qi':qi, 'beta':self.beta}
    
    def predict(self, yf=np.array(list())):
        '''
        Predict the quantile.
        INPUTS:
            - yf: ndarray, optional
                test data. If yf is not empty, the internal state is updated
                with the last observation.Default is an empty list.
        OUTPUTS:
            - qf: ndarray
                quantile estimate.
        '''
        qf = self.loop(self.beta, yf, self.last_state, pred_mode=True)
        if len(yf) > 0:
            self.last_state = [yf[-1], qf[-1]]
        return qf

class CAViaR_2_2(CAViaR_base):
    '''
    CAViaR regression for quantile estimation - y lags=2, q lags=2.
    '''
    def __init__(self, theta, spec='AS'):
        '''
        Initialization of the CAViaR model.
        INPUTS:
            - theta: float
                quantile level.
            - spec: str, optional
                specification of the model (SAV, AS, GARCH). Default is AS.
        OUTPUTS:
            - None
        '''
        self.theta = theta #Initialize theta

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 5
            self.loop = self.SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 7
            self.loop = self.ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 5
            self.loop = self.GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')

    def SAVloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the SAV model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            q = list()
            q.append( beta[0] + beta[1] * np.abs(q0[0]) + beta[2] * q0[1] ) #In pred_mode, q0 is assumed to be [y[-1], q[-1]]
        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1]*np.abs(y[t-1]) + beta[2]*q[t-1] )

        return np.array(q)

    def ASloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the AS model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            # In pred_mode, q0 is assumed to be [y[-max([p, r]):], ^q[-max([p, r]):]]
            q = list()
            for t in range(2):
                q.append( q0[1][t])
            y = np.concatenate([q0[0], y])
            # Loop
            y_coeff_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_l2 = np.where(y>0, beta[3], beta[4])
            if len(y) > 0:
                for t in range(2, len(y)):
                    q.append(beta[0] +\
                            y_coeff_l1[t-1]*y[t-1] + y_coeff_l2[t-2]*y[t-2] +\
                            beta[5]*q[t-1] + beta[6]*q[t-2] )
            else:
                t = 3
                q.append(beta[0] +\
                        y_coeff_l1[t-1]*y[t-1] + y_coeff_l2[t-2]*y[t-2] +\
                        beta[5]*q[t-1] + beta[6]*q[t-2] )
            q = q[2:]

        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]*2
            # Loop
            y_coeff_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_l2 = np.where(y>0, beta[3], beta[4])
            for t in range(2, len(y)):
                q.append(beta[0] +\
                        y_coeff_l1[t-1]*y[t-1] + y_coeff_l2[t-2]*y[t-2] +\
                        beta[5]*q[t-1] + beta[6]*q[t-2] )
        q = np.array(q)
        return q

    def GARCHloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the GARCH model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            q = list()
            q.append( -np.sqrt(beta[0] + beta[1]*q0[0]**2 + beta[2]*q0[1]**2) ) #In pred_mode, q0 is assumed to be [y[-1], q[-1]]
        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0] + beta[1]*y[t-1]**2 + beta[2]*q[t-1]**2) )
        return q

    def fit(self, yi, seed=None, return_train=False, q0=None):
        '''
        Fit the CAViaR model.
        INPUTS:
            - yi: ndarray
                training data.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                return the training prediction. Default is False.
            - q0: float, optional
                initial quantile. Default is None.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - beta: ndarray
                fitted coefficients of the regression (if return_train=True).
        '''
        
        qi = super().fit(yi, seed=seed, q0=q0) #Train CAViaR and recover fitted quantiles
        self.last_state = [yi[-2:], qi[-2:]] #Store the last state

        if return_train: #If return_train is True, return the training prediction and coefficients
            return {'qi':qi, 'beta':self.beta}
    
    def predict(self, yf=np.array(list())):
        '''
        Predict the quantile.
        INPUTS:
            - yf: ndarray, optional
                test data. If yf is not empty, the internal state is updated
                with the last observation.Default is an empty list.
        OUTPUTS:
            - qf: ndarray
                quantile estimate.
        '''
        qf = self.loop(self.beta, yf, self.last_state, pred_mode=True)
        if len(yf) > 0:
            self.last_state = [
                np.concatenate([self.last_state[0], yf])[-2:],
                np.concatenate([self.last_state[1], qf])[-2:] ]
        return qf

class CAViaR_1_2(CAViaR_base):
    '''
    CAViaR regression for quantile estimation - y lags=1, q lags=2.
    '''
    def __init__(self, theta, spec='AS'):
        '''
        Initialization of the CAViaR model.
        INPUTS:
            - theta: float
                quantile level.
            - spec: str, optional
                specification of the model (SAV, AS, GARCH). Default is AS.
        OUTPUTS:
            - None
        '''
        self.theta = theta #Initialize theta

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 4
            self.loop = self.SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 5
            self.loop = self.ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 4
            self.loop = self.GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')

    def SAVloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the SAV model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            q = list()
            q.append( beta[0] + beta[1] * np.abs(q0[0]) + beta[2] * q0[1] ) #In pred_mode, q0 is assumed to be [y[-1], q[-1]]
        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1]*np.abs(y[t-1]) + beta[2]*q[t-1] )

        return np.array(q)

    def ASloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the AS model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            # In pred_mode, q0 is assumed to be [y[-max([p, r]):], ^q[-max([p, r]):]]
            q = list()
            for t in range(2):
                q.append( q0[1][t])
            y = np.concatenate([q0[0], y])
            # Loop
            y_coeff_l1 = np.where(y>0, beta[1], beta[2])
            if len(y) > 0:
                for t in range(2, len(y)):
                    q.append(beta[0] + y_coeff_l1[t-1]*y[t-1] + beta[3]*q[t-1] + beta[4]*q[t-2])
            else:
                t = 3
                q.append(beta[0] + y_coeff_l1[t-1]*y[t-1] + beta[3]*q[t-1] + beta[4]*q[t-2])
            q = q[2:]

        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]*2
            # Loop
            y_coeff_l1 = np.where(y>0, beta[1], beta[2])
            for t in range(2, len(y)):
                q.append(beta[0] + y_coeff_l1[t-1]*y[t-1] + beta[3]*q[t-1] + beta[4]*q[t-2])
        q = np.array(q)
        return q

    def GARCHloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the GARCH model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            q = list()
            q.append( -np.sqrt(beta[0] + beta[1]*q0[0]**2 + beta[2]*q0[1]**2) ) #In pred_mode, q0 is assumed to be [y[-1], q[-1]]
        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0] + beta[1]*y[t-1]**2 + beta[2]*q[t-1]**2) )
        return q

    def fit(self, yi, seed=None, return_train=False, q0=None):
        '''
        Fit the CAViaR model.
        INPUTS:
            - yi: ndarray
                training data.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                return the training prediction. Default is False.
            - q0: float, optional
                initial quantile. Default is None.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - beta: ndarray
                fitted coefficients of the regression (if return_train=True).
        '''
        
        qi = super().fit(yi, seed=seed, q0=q0) #Train CAViaR and recover fitted quantiles
        self.last_state = [yi[-2:], qi[-2:]] #Store the last state

        if return_train: #If return_train is True, return the training prediction and coefficients
            return {'qi':qi, 'beta':self.beta}
    
    def predict(self, yf=np.array(list())):
        '''
        Predict the quantile.
        INPUTS:
            - yf: ndarray, optional
                test data. If yf is not empty, the internal state is updated
                with the last observation.Default is an empty list.
        OUTPUTS:
            - qf: ndarray
                quantile estimate.
        '''
        qf = self.loop(self.beta, yf, self.last_state, pred_mode=True)
        if len(yf) > 0:
            self.last_state = [
                np.concatenate([self.last_state[0], yf])[-2:],
                np.concatenate([self.last_state[1], qf])[-2:] ]
        return qf

class CAViaR_2_1(CAViaR_base):
    '''
    CAViaR regression for quantile estimation - y lags=2, q lags=1.
    '''
    def __init__(self, theta, spec='AS'):
        '''
        Initialization of the CAViaR model.
        INPUTS:
            - theta: float
                quantile level.
            - spec: str, optional
                specification of the model (SAV, AS, GARCH). Default is AS.
        OUTPUTS:
            - None
        '''
        self.theta = theta #Initialize theta

        # According to the desired model, initialize the number of parameters and the loop
        if spec == 'SAV':
            self.mdl_spec = 'SAV'
            self.n_parameters = 4
            self.loop = self.SAVloop
        elif spec == 'AS':
            self.mdl_spec = 'AS'
            self.n_parameters = 6
            self.loop = self.ASloop
        elif spec == 'GARCH':
            self.mdl_spec = 'GARCH'
            self.n_parameters = 4
            self.loop = self.GARCHloop
        else:
            raise ValueError(f'Specification {spec} not recognized!\nChoose between "SAV", "AS", and "GARCH"')

    def SAVloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the SAV model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            q = list()
            q.append( beta[0] + beta[1] * np.abs(q0[0]) + beta[2] * q0[1] ) #In pred_mode, q0 is assumed to be [y[-1], q[-1]]
        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]

        # Loop
        for t in range(1, len(y)):
            q.append( beta[0] + beta[1]*np.abs(y[t-1]) + beta[2]*q[t-1] )

        return np.array(q)

    def ASloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the AS model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            # In pred_mode, q0 is assumed to be [y[-max([p, r]):], ^q[-max([p, r]):]]
            q = list()
            for t in range(2):
                q.append( q0[1][t])
            y = np.concatenate([q0[0], y])
            # Loop
            y_coeff_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_l2 = np.where(y>0, beta[3], beta[4])
            if len(y) > 0:
                for t in range(2, len(y)):
                    q.append(beta[0] +\
                            y_coeff_l1[t-1]*y[t-1] + y_coeff_l2[t-2]*y[t-2] +\
                            beta[5]*q[t-1] )
            else:
                t = 3
                q.append(beta[0] +\
                        y_coeff_l1[t-1]*y[t-1] + y_coeff_l2[t-2]*y[t-2] +\
                        beta[5]*q[t-1] )
            q = q[2:]

        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]*2
            # Loop
            y_coeff_l1 = np.where(y>0, beta[1], beta[2])
            y_coeff_l2 = np.where(y>0, beta[3], beta[4])
            for t in range(2, len(y)):
                q.append(beta[0] +\
                        y_coeff_l1[t-1]*y[t-1] + y_coeff_l2[t-2]*y[t-2] +\
                        beta[5]*q[t-1] )
        q = np.array(q)
        return q

    def GARCHloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the GARCH model.
        INPUTS:
            - beta: ndarray
                regression coefficients.
            - y: ndarray
                true value.
            - q0: float
                initial quantile.
            - pred_mode: bool, optional
                prediction mode. Default is False.
        OUTPUTS:
            - q: ndarray
                quantile estimate.
        '''
        # Initial point
        if pred_mode: #If we are in prediction mode, we have y and q at step -1 and we need to compute q at step 0
            q = list()
            q.append( -np.sqrt(beta[0] + beta[1]*q0[0]**2 + beta[2]*q0[1]**2) ) #In pred_mode, q0 is assumed to be [y[-1], q[-1]]
        else: #If we are in training mode, we have an approximation of q at step 0
            q = [q0]

        # Loop
        for t in range(1, len(y)):
            q.append( -np.sqrt(beta[0] + beta[1]*y[t-1]**2 + beta[2]*q[t-1]**2) )
        return q
 
    def fit(self, yi, seed=None, return_train=False, q0=None):
        '''
        Fit the CAViaR model.
        INPUTS:
            - yi: ndarray
                training data.
            - seed: int or None, optional
                random seed. Default is None.
            - return_train: bool, optional
                return the training prediction. Default is False.
            - q0: float, optional
                initial quantile. Default is None.
        OUTPUTS:
            - qi: ndarray
                quantile forecast in the training set (if return_train=True).
            - beta: ndarray
                fitted coefficients of the regression (if return_train=True).
        '''
        
        qi = super().fit(yi, seed=seed, q0=q0) #Train CAViaR and recover fitted quantiles
        self.last_state = [yi[-2:], qi[-2:]] #Store the last state

        if return_train: #If return_train is True, return the training prediction and coefficients
            return {'qi':qi, 'beta':self.beta}
    
    def predict(self, yf=np.array(list())):
        '''
        Predict the quantile.
        INPUTS:
            - yf: ndarray, optional
                test data. If yf is not empty, the internal state is updated
                with the last observation.Default is an empty list.
        OUTPUTS:
            - qf: ndarray
                quantile estimate.
        '''
        qf = self.loop(self.beta, yf, self.last_state, pred_mode=True)
        if len(yf) > 0:
            self.last_state = [
                np.concatenate([self.last_state[0], yf])[-2:],
                np.concatenate([self.last_state[1], qf])[-2:] ]
        return qf

def CAViaR(theta, spec='AS', p=1, u=1):
    '''
    CAViaR model selection.
    INPUTS:
        - theta: float
            quantile level.
        - spec: str, optional
            specification of the model (SAV, AS, GARCH). Default is AS.
        - p: int, optional
            number of lags for the y variable. Default is 1.
        - u: int, optional
            number of lags for the quantile. Default is 1.
    OUTPUTS:
        - CAViaR model
    '''
    p, u = int(p), int(u) #Convert to integer
    # Ensure that p and r are greater than 0
    assert p > 0, 'p must be greater than 0'
    assert u > 0, 'u must be greater than 0'
    # Check for optimized versions
    if p == 1:
        if u == 1:
            return CAViaR_1_1(theta, spec)
        elif u == 2:
            return CAViaR_1_2(theta, spec)
    elif p == 2:
        if u == 1:
            return CAViaR_2_1(theta, spec)
        elif u == 2:
            return CAViaR_2_2(theta, spec)
    # Otherwise, use thee general version and allert the user
    from warnings import warn
    warnings.warn('The selected model is not optimized. Consider using the optimized versions for p and u in {1,2}.')
    return CAViaR_general(theta, spec, p, u)
