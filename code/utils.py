
import numpy as np

class PinballLoss():
    '''
    Pinball or Quantile loss function
    '''
    def __init__(self, quantile):
        '''
        INPUT:
        - quantile: float,
            the quantile to compute the loss function
        '''
        super().__init__()
        self.quantile = quantile
    
    def __call__(self, y_pred, y_true):
        '''
        INPUT:
        - y_pred: numpy array,
            the predicted values
        - y_true: numpy array,
            the true values
        OUTPUT:
        - loss: float,
            the mean pinball loss
        '''
        #Check consistency in the dimensions
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1,1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1,1)
        if y_pred.shape != y_true.shape:
            raise ValueError(f'Dimensions of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) do not match!!!')
        # Compute the pinball loss
        error = y_true - y_pred
        loss = np.where(error >= 0, self.quantile * error, (self.quantile - 1) * error)
        loss = np.mean(loss)
        return loss

class barrera_loss():
    '''
    Barrera loss function
    '''
    def __init__(self, theta, ret_mean=True):
        '''
        INPUT:
        - theta: float,
            the threshold for the loss function
        - ret_mean: bool,
            if True, the function returns the mean of the loss. Default is True
        '''
        self.theta = theta
        self.ret_mean = ret_mean
    
    def __call__(self, v_, e_, y_):
        '''
        INPUT:
        - v_: numpy array,
            the quantile estimate
        - e_: numpy array,
            the expected shortfall estimate
        - y_: numpy array,
            the actual time series
        OUTPUT:
        - loss: float,
            the loss function mean value, if ret_mean is True. Otherwise, the loss for each observation
        '''
        v, e, y = v_.flatten(), e_.flatten(), y_.flatten()
        r = e - v #Barrera loss is computed on the difference ES - VaR
        if self.ret_mean: #If true, return the mean of the loss
            loss = np.nanmean( (r - np.where(y<v, (y-v)/self.theta, 0))**2 )
        else: #Otherwise, return the loss for each observation
            loss = (r - np.where(y<v, (y-v)/self.theta, 0))**2
        return loss

class patton_loss():
    '''
    Patton loss function
    '''
    def __init__(self, theta, ret_mean=True):
        '''
        INPUT:
        - theta: float,
            the threshold for the loss function
        - ret_mean: bool,
            if True, the function returns the mean of the loss. Default is True
        '''
        self.theta = theta
        self.ret_mean = ret_mean
    
    def __call__(self, v_, e_, y_):
        '''
        INPUT:
        - v_: numpy array,
            the quantile estimate
        - e_: numpy array,
            the expected shortfall estimate
        - y_: numpy array,
            the actual time series
        OUTPUT:
        - loss: float,
            the loss function mean value, if ret_mean is True. Otherwise, the loss for each observation
        '''
        v, e, y = v_.flatten()*100, e_.flatten()*100, y_.flatten()*100
        if self.ret_mean: #If true, return the mean of the loss
            loss = np.nanmean(
                np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e) - 1
            )
        else: #Otherwise, return the loss for each observation
            loss = np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e) - 1
        return loss

class DMtest():
    '''
    Diebold-Mariano test for the equality of forecast accuracy.
    '''
    def __init__(self, loss_func, h = 1):
        '''
        INPUT:
        - loss_func: callable,
            the loss function to compute the forecast accuracy
        - h: int,
            the maximum lag to compute the autocovariance. Default is 1
        '''
        self.loss_func = loss_func
        self.h = h

    def autocovariance(self, Xi, T, k, Xs):
        '''
        Compute the autocovariance of a time series
        INPUT:
        - Xi: numpy array,
            the time series
        - T: int,
            the length of the time series
        - k: int,
            the lag
        - Xs: float,
            the mean of the time series
        OUTPUT:
        - autoCov: float,
            the autocovariance
        '''
        autoCov = 0
        for i in np.arange(0, T-k):
            autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        autoCov = (1/T)*autoCov
        return autoCov
    
    def __call__(self, Q1, E1, Q2, E2, Y):
        '''
        INPUT:
        - Q1: numpy array,
            the first set of quantile predictions
        - E1: numpy array,
            the first set of expected shortfall predictions
        - Q2: numpy array,
            the second set of quantile predictions
        - E2: numpy array,
            the second set of expected shortfall predictions
        - Y: numpy array,
            the actual time series
        OUTPUT:
        - stat: float,
            the test statistic
        - p_value: float,
            the p-value of the test
        - mean_difference: float,
            the mean difference of the losses
        '''
        import warnings
        from scipy.stats import t

        #Compute losses
        e1_lst = self.loss_func(Q1.flatten(), E1.flatten(), Y.flatten())
        e2_lst = self.loss_func(Q2.flatten(), E2.flatten(), Y.flatten())
        d_lst  = e1_lst - e2_lst
        # Clean NaN values, if any
        n = len(d_lst)
        d_lst = d_lst[~np.isnan(d_lst)]
        T = len(d_lst)
        if T < n:
            warnings.warn('There are NaN in the population! They have been removed.', UserWarning)
        if T == 0:
            warnings.warn('All values are NaN!', UserWarning)
            if np.sum(np.isnan(e1_lst)) == n:
                return {'stat':np.nan, 'p_value':0, 'mean_difference':np.inf}
            if np.sum(np.isnan(e2_lst)) == n:
                return {'stat':np.nan, 'p_value':0, 'mean_difference':-np.inf}
            else:
                return {'stat':np.nan, 'p_value':0, 'mean_difference':0}
        else:
            mean_d = np.mean(d_lst)
            
            # Find autocovariance and construct DM test statistics
            gamma = list()
            for lag in range(0, self.h):
                gamma.append(self.autocovariance(d_lst, T, lag, mean_d))
            V_d = (gamma[0] + 2*np.sum(gamma[1:]))/T
            DM_stat = mean_d / np.sqrt(V_d)
            harvey_adj = np.sqrt( (T+1-2*self.h + self.h*(self.h-1)/T) / T )
            DM_stat *= harvey_adj

            # Find p-value
            p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
            
            return {'stat':DM_stat, 'p_value':p_value, 'mean_difference':mean_d}

def cr_t_test(errorsA, errorsB, train_len, test_len):
    '''
    Corrected resampled t-test for the equality of forecast accuracy.
    INPUT:
    - errorsA: numpy array,
        the first set of forecast errors
    - errorsB: numpy array,
        the second set of forecast errors
    - train_len: int,
        the length of the training set
    - test_len: int,
        the length of the test set
    OUTPUT:
    - stat: float,
        the test statistic
    - p_val: float,
        the p-value of the test
    '''
    from scipy.stats import t as stud_t
    output = dict() #Initialize output
    J = len(errorsA) #Compute the number of folds
    if J != len(errorsB):
        raise ValueError('Both samples must have the same length!')
    if isinstance(errorsA, list):
        errorsA = np.array(errorsA)
    if isinstance(errorsB, list):
        errorsB = np.array(errorsB)
    mu_j = errorsA - errorsB #Vector of difference of generalization errors
    mu_hat = np.mean(mu_j) #Mean of the difference of generalization errors
    S2 = np.sum( (mu_j-mu_hat)**2 ) / (J-1) #In sample variance
    sigma2 = (1/J + test_len/train_len)*S2 #Adjusted variance
    return {'stat':mu_hat / np.sqrt(sigma2), 'p_value':stud_t.cdf(output['stat'], J-1)}

class bootstrap_mean_test():
    '''
    Bootstrap test for the mean of a sample
    '''
    def __init__(self, mu_target, one_side=False, n_boot=10_000):
        '''
        INPUT:
        - mu_target: float,
            the mean to test against
        - one_side: bool,
            if True, the test is one sided (i.e. H0: mu >= mu_target). Default is False
        - n_boot: int,
            the number of bootstrap replications. Default is 10_000
        '''
        self.mu_target = mu_target
        self.one_side = one_side
        self.n_boot = n_boot
    
    def null_statistic(self, B_data):
        '''
        Compute the null statistic for the bootstrap sample
        INPUT:
        - B_data: numpy array,
            the bootstrap sample
        OUTPUT:
        - stat: float,
            the null statistic
        '''
        return (np.mean(B_data) - self.obs_mean) * np.sqrt(self.n) / np.std(B_data)
    
    def statistic(self, data):
        '''
        Compute the test statistic for the original sample
        INPUT:
        - data: numpy array,
            the original sample
        OUTPUT:
        - stat: float,
            the test statistic
        '''
        return (self.obs_mean - self.mu_target) * np.sqrt(self.n) / np.std(data)
    
    def __call__(self, data, seed=None):
        '''
        Compute the test
        INPUT:
        - data: numpy array,
            the original sample
        - seed: int,
            the seed for the random number generator. Default is None
        OUTPUT:
        - statistic: float,
            the test statistic
        - p_value: float,
            the p-value of the test
        '''
        np.random.seed(seed)

        self.obs_mean = np.mean(data)
        self.n = len(data)

        B_stats = list()
        for _ in range(self.n_boot):
            B_stats.append( self.null_statistic(
                np.random.choice(data, size=self.n, replace=True) ))
        B_stats = np.array(B_stats)
        self.B_stats = B_stats
        
        if self.one_side:
            obs = self.statistic(data)
            return {'statistic':obs, 'p_value':np.mean(B_stats < obs)}
        else:
            obs = np.abs(self.statistic(data))
            return {'statistic':self.statistic(data),
                    'p_value':np.mean((B_stats > obs) | (B_stats < -obs))}

class McneilFrey_test(bootstrap_mean_test):
    '''
    McNeil-Frey test for assessing the goodness of the Expected Shortfall estimate
    '''
    def __init__(self, one_side=False, n_boot=10_000):
        '''
        INPUT:
        - one_side: bool,
            if True, the test is one sided (i.e. H0: mu >= mu_target). Default is False
        - n_boot: int,
            the number of bootstrap replications. Default is 10_000
        '''
        super().__init__(0, one_side, n_boot)
    
    def mnf_transform(self, Q_, E_, Y_):
        '''
        Transform the data to compute the McNeil-Frey test
        INPUT:
        - Q_: numpy array,
            the quantile estimates
        - E_: numpy array,
            the expected shortfall estimates
        - Y_: numpy array,
            the actual time series
        OUTPUT:
        - output: numpy array,
            the transformed data
        '''
        import warnings

        Q, E, Y = Q_.flatten(), E_.flatten(), Y_.flatten() #Flatten the data
        output = (Y - E)[Y <= Q]
        n = len(output)
        output = output[~np.isnan(output)]
        if len(output) < n:
            warnings.warn('There are NaN in the population! They have been removed.', UserWarning)
        return output

    def __call__(self, Q, E, Y, seed=None):
        '''
        Compute the test
        INPUT:
        - Q: numpy array,
            the quantile estimates
        - E: numpy array,
            the expected shortfall estimates
        - Y: numpy array,
            the actual time series
        - seed: int,
            the seed for the random number generator. Default is None
        OUTPUT:
        - statistic: float,
            the test statistic
        - p_value: float,
            the p-value of the test
        '''
        return super().__call__( self.mnf_transform(Q, E, Y).flatten(), seed)

class AS14_test(bootstrap_mean_test):
    '''
    Acerbi-Szekely test for assessing the goodness of the Expected Shortfall estimate
    '''
    def __init__(self, one_side=False, n_boot=10_000):
        '''
        INPUT:
        - one_side: bool,
            if True, the test is one sided (i.e. H0: mu >= mu_target). Default is False
        - n_boot: int,
            the number of bootstrap replications. Default is 10_000
        '''
        super().__init__(-1, one_side, n_boot)
    
    def as14_transform(self, test_type, Q_, E_, Y_, theta):
        '''
        Transform the data to compute the Acerbi-Szekely test
        INPUT:
        - test_type: str,
            the type of test to perform. It must be either 'Z1' or 'Z2'
        - Q_: numpy array,
            the quantile estimates
        - E_: numpy array,
            the expected shortfall estimates
        - Y_: numpy array,
            the actual time series
        - theta: float,
            the threshold for the test
        OUTPUT:
        - output: numpy array,
            the transformed data
        '''
        import warnings

        Q, E, Y = Q_.flatten(), E_.flatten(), Y_.flatten() #Flatten the data
        if test_type == 'Z1':
            output = (- Y/E)[Y <= Q]
        elif test_type == 'Z2':
            output = - Y * (Y <= Q) / (theta * E)
        else:
            raise ValueError(f'test_type {test_type} not recognized. It must be either Z1 or Z2')
        n = len(output)
        output = output[~np.isnan(output)]
        if len(output) < n:
            warnings.warn('There are NaN in the population! They have been removed.', UserWarning)
        return output

    def __call__(self, Q, E, Y, theta, test_type='Z1', seed=None):
        '''
        Compute the test
        INPUT:
        - Q: numpy array,
            the quantile estimates
        - E: numpy array,
            the expected shortfall estimates
        - Y: numpy array,
            the actual time series
        - test_type: str,
            the type of test to perform. It must be either 'Z1' or 'Z2'
        - theta: float,
            the threshold for the test
        - seed: int,
            the seed for the random number generator. Default is None
        OUTPUT:
        - statistic: float,
            the test statistic
        - p_value: float,
            the p-value of the test
        '''
        return super().__call__( self.as14_transform(test_type, Q, E, Y, theta).flatten(), seed)

class LossDiff_test(bootstrap_mean_test):
    '''
    Encompassing test to assess whenever the first sample of losses is statistically lower than the second.
    '''
    def __init__(self, loss, n_boot=10_000):
        '''
        INPUT:
        - loss: callable,
            the loss function to compute the forecast accuracy
        - n_boot: int,
            the number of bootstrap replications. Default is 10_000
        '''
        super().__init__(0, True, n_boot)
        self.loss = loss
    
    def ld_transform(self, Q_new, E_new, Q_bench, E_bench, Y):
        '''
        Transform the data to compute the test
        INPUT:
        - Q_new: numpy array,
            the first set of quantile predictions
        - E_new: numpy array,
            the first set of expected shortfall predictions
        - Q_bench: numpy array,
            the second set of quantile predictions
        - E_bench: numpy array,
            the second set of expected shortfall predictions
        - Y: numpy array,
            the actual time series
        OUTPUT:
        - output: numpy array,
            the transformed data
        '''
        import warnings
        output = self.loss(Q_new, E_new, Y) - self.loss(Q_bench, E_bench, Y)
        n = len(output)
        output = output[~np.isnan(output)]
        if len(output) < n:
            warnings.warn('There are NaN in the population! They have been removed.', UserWarning)
        return output
    
    def __call__(self, Q_new, E_new, Q_bench, E_bench, Y, seed=None):
        '''
        Compute the test
        INPUT:
        - Q_new: numpy array,
            the first set of quantile predictions
        - E_new: numpy array,
            the first set of expected shortfall predictions
        - Q_bench: numpy array,
            the second set of quantile predictions
        - E_bench: numpy array,
            the second set of expected shortfall predictions
        - Y: numpy array,
            the actual time series
        - seed: int,
            the seed for the random number generator. Default is None
        OUTPUT:
        - statistic: float,
            the test statistic
        - p_value: float,
            the p-value of the test
        '''
        return super().__call__( self.ld_transform(
            Q_new, E_new, Q_bench, E_bench, Y).flatten(), seed)

class Encompassing_test(bootstrap_mean_test):
    '''
    Encompassing test to assess whenever the first sample of losses is statistically lower than the second.
    '''
    def __init__(self, loss, n_boot=10_000):
        '''
        INPUT:
        - loss: callable,
            the loss function to compute the forecast accuracy
        - n_boot: int,
            the number of bootstrap replications. Default is 10_000
        '''
        super().__init__(0, True, n_boot)
        self.loss = loss

    def en_transform(self, Q_new_, E_new_, Q_bench_, E_bench_, Y_):
        '''
        Transform the data to compute the test
        INPUT:
        - Q_new_: numpy array,
            the first set of quantile predictions
        - E_new_: numpy array,
            the first set of expected shortfall predictions
        - Q_bench_: numpy array,
            the second set of quantile predictions
        - E_bench_: numpy array,
            the second set of expected shortfall predictions
        - Y_: numpy array,
            the actual time series
        OUTPUT:
        - output: numpy array,
            the transformed data
        '''
        import warnings
        from scipy.optimize import minimize

        # Flatten the arrays
        Q_new, E_new, Q_bench, E_bench, Y = Q_new_.flatten(), E_new_.flatten(), Q_bench_.flatten(), E_bench_.flatten(), Y_.flatten()

        # Split into train and test sets
        train_size = Q_new.shape[0]//2
        Q_new_train, E_new_train = Q_new[:train_size], E_new[:train_size]
        Q_new_test, E_new_test = Q_new[train_size:], E_new[train_size:]
        Q_bench_train, E_bench_train = Q_bench[:train_size], E_bench[:train_size]
        Q_bench_test, E_bench_test = Q_bench[train_size:], E_bench[train_size:]
        Y_train, Y_test = Y[:train_size], Y[train_size:]

        #Fit the linear model
        bounds = [(0,1), (0,1)]
        alpha = minimize(lambda x: np.nanmean(self.loss(
            Q_new_train*x[0] + Q_bench_train*x[1],
            E_new_train*x[0] + E_bench_train*x[1], Y_train)),
                        [0.5, 0.5], bounds=bounds, method='SLSQP',
                        options={'disp': False}, tol=1e-6).x
        
        # Compute the population
        output = self.loss(Q_new_test, E_new_test, Y_test) - self.loss(
            Q_new_test*alpha[0] + Q_bench_test*alpha[1],
            E_new_test*alpha[0] + E_bench_test*alpha[1], Y_test)
        n = len(output)
        output = output[~np.isnan(output)]
        if len(output) < n:
            warnings.warn('There are NaN in the population! They have been removed.', UserWarning)
        return output
        
    def __call__(self, Q_new, E_new, Q_bench, E_bench, Y, seed=None):
        '''
        Compute the test
        INPUT:
        - Q_new: numpy array,
            the first set of quantile predictions
        - E_new: numpy array,
            the first set of expected shortfall predictions
        - Q_bench: numpy array,
            the second set of quantile predictions
        - E_bench: numpy array,
            the second set of expected shortfall predictions
        - Y: numpy array,
            the actual time series
        - seed: int,
            the seed for the random number generator. Default is None
        OUTPUT:
        - statistic: float,
            the test statistic
        - p_value: float,
            the p-value of the test
        '''
        return super().__call__( self.en_transform(
            Q_new, E_new, Q_bench, E_bench, Y).flatten(), seed)
    
def gaussian_tail_stats(theta, loc=0, scale=1):
    '''
    Compute the Value at Risk and Expected Shortfall for a Gaussian distribution
    INPUT:
    - theta: float,
        the quantile to compute the statistics
    - loc: numpy array,
        the mean of the distribution
    - scale: numpy array,
        the standard deviation of the distribution
    OUTPUT:
    - var: numpy array,
        the Value at Risk
    - es: numpy array,
        the Expected Shortfall
    '''
    from scipy.stats import norm
    # If working with scalar, convert to numpy array
    if isinstance(loc, (int, float)):
        loc = np.array([loc])
    if isinstance(scale, (int, float)):
        scale = np.array([scale])
    
    # Raise error if the dimensions do not match
    if loc.shape != scale.shape:
        raise ValueError(f'loc and scale must have the same dimensions!\nFound loc={loc.shape} and scale={scale.shape}')

    # Compute the Expected Shortfall
    var = np.zeros_like(loc)
    es = np.zeros_like(loc)
    for t in range(len(loc)):
        es[t] = loc[t] - scale[t]*norm.pdf(norm.ppf(1-theta))/theta
        var[t] = loc[t] + scale[t]*norm.ppf(theta)
    # If the input was a scalar, return scalars
    if len(var) == 1:
        return {'var':var[0], 'es':es[0]}
    else:
        return {'var':var, 'es':es}

def tstudent_tail_stats(theta, df, loc=0, scale=1):
    '''
    Compute the Value at Risk and Expected Shortfall for a Student's t distribution
    INPUT:
    - theta: float,
        the quantile to compute the statistics
    - df: int,
        the degrees of freedom of the distribution
    - loc: numpy array,
        the mean of the distribution
    - scale: numpy array,
        the standard deviation of the distribution
    OUTPUT:
    - var: numpy array,
        the Value at Risk
    - es: numpy array,
        the Expected Shortfall
    '''
    from scipy.stats import t as t_dist
    from scipy.special import gamma as gamma_func

    # If working with scalar, convert to numpy array
    if isinstance(loc, (int, float)):
        loc = np.array([loc])
    if isinstance(scale, (int, float)):
        scale = np.array([scale])
    
    # Raise error if the dimensions do not match
    if loc.shape != scale.shape:
        raise ValueError('loc and scale must have the same dimensions!')

    # Compute the Expected Shortfall
    cte = gamma_func((df+1)/2) / (np.sqrt(np.pi*df)*gamma_func(df/2))
    var = np.zeros_like(loc)
    es = np.zeros_like(loc)
    for t in range(len(loc)):
        var[t] = t_dist.ppf(theta, df=df, loc=0, scale=1)
        tau = cte * (1 + var[t]**2/df)**(-(1+df)/2)
        es[t] = loc[t] - scale[t] * (df + var[t]**2) * tau / ( (df-1) * theta)
        var[t] = loc[t] + var[t] * scale[t]
    # If the input was a scalar, return scalars
    if len(var) == 1:
        return {'var':var[0], 'es':es[0]}
    else:
        return {'var':var, 'es':es}
