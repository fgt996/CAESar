
import numpy as np

class barrera_loss():
    def __init__(self, theta, ret_mean=True):
        self.theta = theta
        self.ret_mean = ret_mean
    
    def __call__(self, v_, e_, y_):
        v, e, y = v_.flatten(), e_.flatten(), y_.flatten()
        r = e - v
        if self.ret_mean:
            return np.nanmean(
                (r - np.where(y<v, (y-v)/self.theta, 0))**2
                )
        else:
            return (r - np.where(y<v, (y-v)/self.theta, 0))**2

class patton_loss():
    def __init__(self, theta, ret_mean=True):
        self.theta = theta
        self.ret_mean = ret_mean
    
    def __call__(self, v_, e_, y_):
        v, e, y = v_.flatten()*100, e_.flatten()*100, y_.flatten()*100
        if self.ret_mean:
            return np.nanmean(
                np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e) - 1
            )
        else:
            return np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e) - 1

class DBtest():
    def __init__(self, loss_func, h = 1):
        self.loss_func = loss_func
        self.h = h

    def autocovariance(self, Xi, T, k, Xs):
        autoCov = 0
        for i in np.arange(0, T-k):
            autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/T)*autoCov
    
    def __call__(self, Q1, E1, Q2, E2, Y):
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
    output['statistic'] = mu_hat / np.sqrt(sigma2) #Compute test statistic
    output['p_val'] = stud_t.cdf(output['statistic'], J-1) #Compute the p-value
    return output

class bootstrap_mean_test():
    def __init__(self, mu_target, one_side=False, n_boot=10000):
        '''
        INPUT:
        - mu_target: float, the mean to test against
        - one_side: bool, if True, the test is one sided (i.e. H0: mu >= mu_target)
        - n_boot: int, the number of bootstrap replications
        '''
        self.mu_target = mu_target
        self.one_side = one_side
        self.n_boot = n_boot
    
    def null_statistic(self, B_data):
        return (np.mean(B_data) - self.obs_mean) * np.sqrt(self.n) / np.std(B_data)
    
    def statistic(self, data):
        return (self.obs_mean - self.mu_target) * np.sqrt(self.n) / np.std(data)
    
    def __call__(self, data, seed=None):
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
    def __init__(self, one_side=False, n_boot=10000):
        super().__init__(0, one_side, n_boot)
    
    def mnf_transform(self, Q_, E_, Y_):
        import warnings

        Q, E, Y = Q_.flatten(), E_.flatten(), Y_.flatten() #Flatten the data
        output = (Y - E)[Y <= Q]
        n = len(output)
        output = output[~np.isnan(output)]
        if len(output) < n:
            warnings.warn('There are NaN in the population! They have been removed.', UserWarning)
        return output

    def __call__(self, Q, E, Y, seed=None):
        return super().__call__( self.mnf_transform(Q, E, Y).flatten(), seed)

class AS14_test(bootstrap_mean_test):
    def __init__(self, one_side=False, n_boot=10000):
        super().__init__(-1, one_side, n_boot)
    
    def as14_transform(self, test_type, Q_, E_, Y_, theta=None):
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

    def __call__(self, Q, E, Y, test_type='Z1', theta=None, seed=None):
        return super().__call__( self.as14_transform(test_type, Q, E, Y, theta).flatten(), seed)

class LossDiff_test(bootstrap_mean_test):
    def __init__(self, loss, n_boot=10000):
        super().__init__(0, True, n_boot)
        self.loss = loss
    
    def ld_transform(self, Q_new, E_new, Q_bench, E_bench, Y):
        import warnings
        temp = self.loss(Q_new, E_new, Y) - self.loss(Q_bench, E_bench, Y)
        n = len(temp)
        temp = temp[~np.isnan(temp)]
        if len(temp) < n:
            warnings.warn('There are NaN in the population! They have been removed.', UserWarning)
        return temp
    
    def __call__(self, Q_new, E_new, Q_bench, E_bench, Y, seed=None):
        return super().__call__( self.ld_transform(
            Q_new, E_new, Q_bench, E_bench, Y).flatten(), seed)

class Encompassing_test(bootstrap_mean_test):
    def __init__(self, loss, n_boot=10000):
        super().__init__(0, True, n_boot)
        self.loss = loss

    def en_transform(self, Q_new_, E_new_, Q_bench_, E_bench_, Y_):
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
        population = self.loss(Q_new_test, E_new_test, Y_test) - self.loss(
            Q_new_test*alpha[0] + Q_bench_test*alpha[1],
            E_new_test*alpha[0] + E_bench_test*alpha[1], Y_test)
        n = len(population)
        population = population[~np.isnan(population)]
        if len(population) < n:
            warnings.warn('There are NaN in the population! They have been removed.', UserWarning)
        return population
        
    def __call__(self, Q_new, E_new, Q_bench, E_bench, Y, seed=None):
        return super().__call__( self.en_transform(
            Q_new, E_new, Q_bench, E_bench, Y).flatten(), seed)
    