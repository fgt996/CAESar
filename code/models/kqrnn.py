
import copy
import torch
import numpy as np
import torch.nn as nn

# Define the base neural network class
class BaseNN():
    def __init__(self):
        self.set_activation_function() # Define the activation function
        self.set_regularizer() # Define the regularization
    
    # Set the activation function according to the parameter 'activation' (default: tanh)
    def set_activation_function(self):
        #Three available activation functions: tanh, relu, sigmoid
        temp = self.params['activation'].lower() if type(self.params['activation']) == str else None
        if temp == 'tanh':
            self.activation = nn.Tanh()
        elif temp == 'relu':
            self.activation = nn.ReLU()
        elif temp == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Activation function {self.params['activation']} not recognized")
    
    # Set the regularization according to the parameter 'reg_type' (default: no regularization)
    def set_regularizer(self):
        #Two available regularization types: l1, l2, and l1_l2
        self.reg = torch.tensor(self.params['reg']).to(self.dev)
        #Eventually, apply l1 regularization
        if self.params['reg_type'] == 'l1':
            def l1_model_reg(model):
                regularization_loss = torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    regularization_loss += torch.norm(param, 1)
                return self.reg*regularization_loss
            self.regularizer = l1_model_reg
        #Eventually, apply l2 regularization
        elif self.params['reg_type'] == 'l2':
            def l2_model_reg(model):
                regularization_loss = torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    regularization_loss += torch.norm(param, 2)
                return self.reg*regularization_loss
            self.regularizer = l2_model_reg
        #Eventually, apply l1_l2 regularization
        elif self.params['reg_type'] == 'l1_l2':
            def l1_l2_model_reg(model):
                l1_loss, l2_loss = torch.tensor(0.0).to(self.dev), torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    l1_loss += torch.norm(param, 1)
                    l2_loss += torch.norm(param, 2)
                return self.reg[0]*l1_loss + self.reg[1]*l2_loss
            self.regularizer = l1_l2_model_reg
        #Eventually, no regularization is applied
        else:
            def no_reg(model):
                return torch.tensor(0.0)
            self.regularizer = no_reg
        
    def set_optimizer(self):
        #Two available optimizers: Adam, RMSprop
        temp = self.params['optimizer'].lower() if type(self.params['optimizer']) == str else None
        if temp.lower() == 'adam':
            from torch.optim import Adam
            self.opt = Adam(self.mdl.parameters(), self.params['lr'])
        elif temp.lower() == 'rmsprop':
            from torch.optim import RMSprop
            self.opt = RMSprop(self.mdl.parameters(), self.params['lr'])
        else:
            raise ValueError(f"Optimizer {temp} not recognized")
    
    # Training with full batch
    def train_single_batch(self, x_train, y_train):
        self.mdl.train() #Set the model in training mode
        self.opt.zero_grad() # Zero the gradients
        outputs = self.mdl(x_train) # Forward pass
        loss = self.loss(outputs, y_train) + self.regularizer(self.mdl)
        loss.backward()  # Backpropagation
        self.opt.step()  # Update weights
        self.train_loss.append(loss.item()) #Save the training loss
    
    # Training with multiple batches
    def train_multi_batch(self, x_train, y_train, indices):
        self.mdl.train() #Set the model in training mode
        #Prepare batch training
        total_loss = 0.0
        indices = indices[torch.randperm(indices.size(0))] #Shuffle the indices
        # Training
        for i in range(0, len(indices), self.params['batch_size']):
            #Construct the batch
            batch_indices = indices[i:i+self.params['batch_size']] #Select the indices
            x_batch = x_train[batch_indices] #Select the batch
            y_batch = y_train[batch_indices]

            self.opt.zero_grad() # Zero the gradients
            outputs = self.mdl(x_batch) # Forward pass
            loss = self.loss(outputs, y_batch) + self.regularizer(self.mdl)
            loss.backward()  # Backpropagation
            self.opt.step()  # Update weights
            total_loss += loss.item()
        total_loss /= np.ceil(len(indices) / self.params['batch_size'])
        self.train_loss.append(total_loss) #Save the training loss
    
    def early_stopping(self, curr_loss):
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss #Save the best loss
            self.best_model = copy.deepcopy(self.mdl.state_dict()) #Save the best model
            self.patience = 0 #Reset the patience counter
        else:
            self.patience += 1
        #Check if I have to exit
        if self.patience > self.params['patience']:
            return True
        else:
            return False

    #Fit the model. With early stopping and batch training
    def fit(self, x_train, y_train, x_val=None, y_val=None):
        #Initialize the best model
        self.best_model = self.mdl.state_dict()
        self.best_loss = np.inf
        #Initialize the patience counter
        self.patience = 0
        #Initialize the training and validation losses
        self.train_loss = []
        if isinstance(x_val, torch.Tensor):
            self.val_loss = []
        else:
            self.val_loss = None
        #Understand if I'm using single or multiple batches
        single_batch = (self.params['batch_size'] == -1) or\
            (self.params['batch_size'] >= x_train.shape[0])
        # Eventually, create the train indices list
        if not single_batch: indices = torch.arange(x_train.shape[0])
        # Set the verbosity if it is provided as a boolean:
        if isinstance(self.verbose, bool):
            self.verbose = int(self.verbose) - 1
        #Train the model
        if self.verbose >= 0: #If verbose is True, then use the progress bar
            from tqdm.auto import tqdm
            it_base = tqdm(range(self.params['n_epochs']), desc='Training the network') #Create the progress bar
        else: #Otherwise, use the standard iterator
            it_base = range(self.params['n_epochs']) #Create the iterator
        for epoch in it_base:
            if (epoch==0) and (self.verbose > 0):
                print_base = '{:<10}{:<15}{:<15}'
                print(print_base.format('Epoch', 'Train Loss', 'Val Loss'))
            # Training
            if single_batch:
                self.train_single_batch(x_train, y_train)
            else:
                self.train_multi_batch(x_train, y_train, indices)
            # If Validation
            if isinstance(x_val, torch.Tensor):
                self.mdl.eval() #Set the model in evaluation mode
                with torch.no_grad():
                    val_loss = self.loss(self.mdl(x_val), y_val) #Compute the validation loss
                    self.val_loss.append(val_loss.item()) #Save the validation loss
                # Update best model and eventually early stopping
                if self.early_stopping(val_loss):
                    if self.verbose >= 0: print(f"Early stopping at epoch {epoch+1}")
                    break
            else: #Otherwise
                # Update best model and eventually early stopping
                if self.early_stopping(self.train_loss[-1]):
                    if self.verbose >= 0: print(f"Early stopping at epoch {epoch+1}")
                    break
            # Eventually, print the losses
            if self.verbose > 0:
                if (epoch+1) % self.verbose == 0:
                    print(print_base.format(epoch+1,
                        format(self.train_loss[-1], '.20f')[:10],
                        format(self.val_loss[-1], '.20f')[:10] if isinstance(x_val, torch.Tensor) else '-'))
        #Load the best model
        self.mdl.load_state_dict(self.best_model)

    #Plot the training and validation losses
    def plot_losses(self, yscale='log'):
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme()
        #Plot the losses
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        ax.set(yscale=yscale)
        sns.lineplot(self.train_loss, label='Train')
        if isinstance(self.val_loss, list):
            sns.lineplot(self.val_loss, label='Validation')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Losses")
        plt.legend()
        plt.show()
    
    #Set the model in evaluation mode
    def eval_mode(self):
        self.mdl.drop = 0

# Define the feedforward neural network class
class FFN(nn.Module):
    def __init__(self, layers, init, a_fun, drop, DTYPE=torch.float64):
        '''
        INPUT:
            - layers: list of integers, where each integer is the number of neurons
                in the corresponding layer
            - init: string, the type of initialization to use for the weights
            - a_fun: activation function
            - drop: dropout rate
            - DTYPE: torch datatype
        OUTPUT:
            - Feedforward neural network
        '''
        super().__init__()
        # Define the layers
        self.layers = nn.ModuleList([
            nn.Linear(layers[l-1], layers[l], dtype=DTYPE) for\
                l in range(1,len(layers))
                ])
        # Initialize the weights
        self.weights_initializer(init)
        #Define activation function and dropout
        self.activation = a_fun
        self.drop = drop
    
    # Initialize the weights, according to the parameter init (default: glorot_normal)
    def weights_initializer(self, init):
        '''
        INPUT:
            - init: string, the type of initialization to use for the weights
        OUTPUT:
            - None
        '''
        #Two available initializers: glorot_normal, glorot_uniform
        temp = init.lower() if type(init) == str else None
        if temp == 'glorot_normal':
            for layer in self.layers:
                nn.init.xavier_normal_(layer.weight)
        elif temp == 'glorot_uniform':
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight)
        else:
            raise ValueError(f"Initializer {init} not recognized")

    def forward(self, x):
        '''
        INPUT:
            - x: torch tensor of shape (batch_size, n_features)
        OUTPUT:
            - torch tensor of shape (batch_size, output_size)
        '''
        # Forward pass through the network
        if self.drop == 0: #Case 1: no dropout
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))
        elif self.drop > 0: #Case 2: dropout
            for layer in self.layers[:-1]:
                x = layer(x)
                x = self.activation(x)
                x = nn.Dropout(p=self.drop)(x)
        else: #Case 3: negative dropout => error!
            raise ValueError(f"Dropout value {self.drop} is negative!")
        x = self.layers[-1](x)
        return x

# Define the loss function class
class PinballLoss_MultiQ(nn.Module):
    def __init__(self, quantiles):
        '''
        INPUT:
            - quantile: float between 0 and 1
        OUTPUT:
            - Pinball loss function
        '''
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, y_pred, y_true):
        '''
        INPUT:
            - y_pred: torch tensor of shape (batch_size, n_series)
            - y_true: torch tensor of shape (batch_size, n_series)
        OUTPUT:
            - Pinball loss
        '''
        # Ensure to work with torch tensors
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        #Check consistency in the dimensions
        if len(y_pred.shape) == 1:
            y_pred = torch.unsqueeze(y_pred, dim=1)
        if len(y_true.shape) == 1:
            y_true = torch.unsqueeze(y_true, dim=1)
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(f'Shape[0] of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) do not match!!!')
        if y_pred.shape[1] != len(self.quantiles):
            raise ValueError(f'Shape[1] of y_pred ({y_pred.shape}) and len(quantiles) ({len(self.quantiles)}) do not match!!!')
        if y_true.shape[1] != 1:
            raise ValueError(f'Shape[1] of y_true ({y_pred.shape}) should be 1!!!')
        # Compute the pinball loss
        error = y_true - y_pred
        loss = torch.zeros(y_true.shape).to(y_true.device)
        for q, quantile in enumerate(self.quantiles):
            loss += torch.max(quantile * error[:,q:q+1], (quantile - 1) * error[:,q:q+1])
        return torch.mean(loss)

# Define the K-QRNN class
class K_QRNN(BaseNN):
    def __init__(self, params, dev, verbose=True):
        '''
        INPUTS:
            - params: dictionary with the parameters of the model
            - dev: device where the model will be trained
            - verbose: if True, print the training progress
        PARAMETERS:
            - input_dim: dimension of the input
            - layers: list of integers, where each integer is the number of neurons
                in the corresponding layer
            - theta: quantile level
            - initializer: string, the type of initialization to use for the weights;
                default: 'glorot_normal'
            - activation: activation function; default: 'relu'
            - dropout: dropout rate; default: 0
            - lr: learning rate; default: 0.01
            - batch_size: batch size; default: -1 (full batch)
            - patience: patience for early stopping; default: np.inf (no early stopping)
        OUTPUTS:
            - None
        '''
        self.set_params(params) #Set the parameters
        self.dev = dev
        self.verbose = verbose
        super().__init__()
        # Define the model and optimizer
        self.mdl = FFN(self.params['layers'], self.params['initializer'],
                       self.activation, self.params['dropout']).to(self.dev)
        self.set_optimizer() #Define the optimizer
        # Define the loss function
        self.loss = PinballLoss_MultiQ(
            np.linspace(0, self.params['theta'], self.params['n_points']+1, endpoint=True)[1:])
    
    def set_params(self, params):
        '''
        Define the ultimate parameters dictionary by merging the parameters
        defined by the user with the default ones
        '''
        self.params = {'optimizer': 'Adam', 'reg_type': None,
                       'reg': 0, 'initializer': 'glorot_normal', 'activation': 'relu',
                       'lr': 0.01, 'dropout': 0, 'batch_size':-1,
                       'patience': np.inf, 'verbose': 1} #Default parameters
        self.params.update(params) #Update default parameters with those provided by the user
    
    def __call__(self, x_test):
        res = self.mdl(x_test)
        return {'qf':res[:,-1].cpu().detach().numpy(),
                'ef':torch.mean(res, dim=1).cpu().detach().numpy()}
    