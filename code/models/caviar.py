
import numpy as np

class CAViaR():
    '''
    CAViaR regression for quantile estimation.
    '''
    def __init__(self, theta, spec='AS'):
        '''
        Initialization of the CAViaR model.
        INPUTS:
            - theta: quantile level
            - spec: specification of the model (SAV, AS, GARCH). Default is AS
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
    
    def loss_function(self, q, y):
        '''
        Pinball loss function.
        INPUTS:
            - q: quantile estimate
            - y: true value
        OUTPUTS:
            - Pinball loss: loss value
        '''
        return np.mean(
            np.where(y<q, (1-self.theta), -self.theta) * (q-y)
        )
    
    def SAVloop(self, beta, y, q0, pred_mode=False):
        '''
        Loop for the SAV model.
        INPUTS:
            - beta: parameters
            - y: true value
            - q0: initial quantile
            - pred_mode: prediction mode. Default is False
        OUTPUTS:
            - q: quantile estimate
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
            - beta: parameters
            - y: true value
            - q0: initial quantile
            - pred_mode: prediction mode. Default is False
        OUTPUTS:
            - q: quantile estimate
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
            - beta: parameters
            - y: true value
            - q0: initial quantile
            - pred_mode: prediction mode. Default is False
        OUTPUTS:
            - q: quantile estimate
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

    def Qloss(self, beta, y, q0):
        '''
        Loss function for the CAViaR model.
        INPUTS:
            - beta: parameters
            - y: true value
            - q0: initial quantile
        OUTPUTS:
            - loss: loss value
        '''
        q = self.loop(beta, y, q0)
        return self.loss_function(q, y)
    
    def fit(self, yi, seed=None, return_train=False, q0=None):
        '''
        Fit the CAViaR model.
        INPUTS:
            - yi: training data
            - seed: random seed. Default is None
            - return_train: return the training prediction. Default is False
            - q0: initial quantile. Default is None
        OUTPUTS:
            - qi: training prediction (if return_train=True)
        '''
        from scipy.optimize import fmin

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

        #-------------------- Step 3: Optimization Routine - I step (barrera)
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
        
        #in sample variables
        qi = self.loop(self.beta, yi, q0)
        self.last_state = [yi[-1], qi[-1]]
        if return_train:
            return {'qi':qi, 'beta':self.beta}
    
    def predict(self, yf=list()):
        '''
        Predict the quantile.
        INPUTS:
            - yf: test data. Default is an empty list
        OUTPUTS:
            - qf: quantile estimate. If yf is not empty, the internal state is updated with the last observation
        '''
        qf = self.loop(self.beta, yf, self.last_state, pred_mode=True)
        if len(yf) > 0:
            self.last_state = [yf[-1], qf[-1]]
        return qf

    def fit_predict(self, y, ti, seed=None, return_train=True, q0=None):
        '''
        Fit the model and predict the quantile.
        INPUTS:
            - y: data
            - ti: train/test split point
            - seed: random seed. Default is None
            - return_train: return the training prediction. Default is True
            - q0: initial quantile. Default is None
        OUTPUTS:
            - dict{'qi': training prediction (if return_train=True),
                'qf': test prediction,
                'beta': model parameters}
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
