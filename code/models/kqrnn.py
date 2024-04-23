
import copy
import torch
import numpy as np
import torch.nn as nn

# Define the base neural network class
class BaseNN():
    '''
    Base class for neural networks.
    '''
    def __init__(self):
        '''
        Initialize the base neural network class.
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
        self.set_regularizer() # Define the regularization
    
    def set_regularizer(self):
        '''
        Set the regularization according to the parameter 'reg_type' (default: no regularization).
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
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
        '''
        Set the optimizer according to the parameter 'optimizer' (default: Adam).
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
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
    
    def train_single_batch(self, x_train, y_train):
        '''
        Training with full batch.
        INPUT:
            - x_train: torch.Tensor
                model's input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's output of shape (batch_size, 1).
        OUTPUT:
            - None.
        '''
        self.mdl.train() #Set the model in training mode
        self.opt.zero_grad() # Zero the gradients
        outputs = self.mdl(x_train) # Forward pass
        loss = self.loss(outputs, y_train) + self.regularizer(self.mdl)
        loss.backward()  # Backpropagation
        self.opt.step()  # Update weights
        self.train_loss.append(loss.item()) #Save the training loss
    
    def train_multi_batch(self, x_train, y_train, indices):
        '''
        Training with multiple batches.
        INPUT:
            - x_train: torch.Tensor
                model's input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's output of shape (batch_size, 1).
            - indices: torch.Tensor
                list of indices (range(batch_size)).
        OUTPUT:
            - None.
        '''
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
        '''
        Early stopping function.
        INPUT:
            - curr_loss: float,
                current loss.
        OUTPUT:
            - output: bool,
                True if early stopping is satisfied, False otherwise.
        '''
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss #Save the best loss
            self.best_model = copy.deepcopy(self.mdl.state_dict()) #Save the best model
            self.patience = 0 #Reset the patience counter
        else:
            self.patience += 1
        #Check if I have to exit
        if self.patience > self.params['patience']:
            output = True
        else:
            output = False
        return output

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        '''
        Fit the model. With early stopping and batch training.
        INPUT:
            - x_train: torch.Tensor
                model's train input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's train output of shape (batch_size, 1).
            - x_val: torch.Tensor, optional
                model's validation input of shape (batch_size, n_features). Default is None.
            - y_val: torch.Tensor, optional
                model's validation output of shape (batch_size, 1). Default is None.
        OUTPUT:
            - None.
        '''
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

    def plot_losses(self, yscale='log'):
        '''
        Plot the training loss and, eventually, the validation loss.
        INPUT:
            - yscale: str, optional
                scale of the y-axis. Default is 'log'.
        OUTPUT:
            - None.
        '''
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

# Define the feedforward neural network class
class FFN(nn.Module):
    '''
    Class for Feedforward neural networks.
    '''
    def __init__(self, layers, init, activation, drop, DTYPE=torch.float64):
        '''
        INPUT:
            - layers: list of int
                list such that each component is the number of neurons in the corresponding layer.
            - init: str
                the type of initialization to use for the weights. Either 'glorot_normal' or 'glorot_uniform'.
            - activation: str
                name of the activation function. Either 'tanh', 'relu', or 'sigmoid'.
            - drop: float
                dropout rate.
            - DTYPE: torch data type.
        OUTPUT:
            - None.
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
        self.set_activation_function(activation) #Define activation function
        self.dropout = nn.Dropout(drop)
    
    def weights_initializer(self, init):
        '''
        Initialize the weights.
        INPUT:
            - init: str
                type of initialization to use for the weights.
        OUTPUT:
            - None.
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

    def set_activation_function(self, activation):
        '''
        Set the activation function.
        INPUT:
            - activation: str
                type of activation function to use.
        OUTPUT:
            - None.
        '''
        #Three available activation functions: tanh, relu, sigmoid
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Activation function {activation} not recognized")
        
    def forward(self, x):
        '''
        INPUT:
            - x: torch.Tensor
                input of the network; shape (batch_size, n_features).
        OUTPUT:
            - output: torch.Tensor
                output of the network; shape (batch_size, output_size).
        '''
        # Forward pass through the network
        for layer in self.layers[:-1]:
            x = self.activation(layer(x)) #Hidden layers
            x = self.dropout(x) #Dropout
        output = self.layers[-1](x) #Output layer
        return output

# Define the loss function class
class PinballLoss_MultiQ(nn.Module):
    '''
    Class for the Pinball loss function.
    '''
    def __init__(self, quantiles):
        '''
        Initialize the Pinball loss function.
        INPUT:
            - quantiles: list of float
                each element is between 0 and 1 and represents a target confidence levels.
        OUTPUT:
            - None.
        '''
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, y_pred, y_true):
        '''
        INPUT:
            - y_pred: torch.Tensor
                quantile forecasts with shape (batch_size, n_series).
            - y_true: torch.Tensor
                actual values with shape (batch_size, n_series).
        OUTPUT:
            - loss: float
                mean pinball loss.
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
        loss = torch.mean(loss)
        return loss

# Define the K-QRNN class
class K_QRNN(BaseNN):
    '''
    Expected Shortfall estimation via Kratz approach with Quantile Regression Neural Network
        (QRNN) for quantile regression.
    '''
    def __init__(self, params, dev, verbose=True):
        '''
        Initialization of the K-QRNN model.
        INPUTS:
            - params: dict
                parameters of the model.
            - dev: torch.device
                indicates the device where the model will be trained.
            - verbose: bool, optional
                if True, print the training progress. Default is True.
        PARAMS:
            - optimizer: str, optional
                optimizer to use, either 'Adam' or 'RMSProp'. Default is 'Adam'.
            - reg_type: str or None, optional
                type of regularization. Either None, 'l1', 'l2', or 'l1_l2'. Default is None.
            - reg: float or list of float, optional
                regularization parameter. Not consider when reg_type=None.
                float when reg_type='l1' or 'l2'. List with two floats (l1 and l2) when
                reg_type='l1_l2'. Default is 0.
            - initializer: str, optional
                initializer for the weights. Either 'glorot_normal' or 'glorot_uniform'.
                Default is 'glorot_normal'.
            - activation: str, optional
                activation function. Either 'relu', 'sigmoid', or 'tanh'. Default is 'relu'.
            - lr: float, optional
                learning rate. Default is 0.01.
            - dropout: float, optional
                dropout rate. Default is 0.
            - batch_size: int, optional
                batch size. Default is -1, that is full batch. When
                batch_size < x_train.shape[0], mini-batch training is performed.
            - patience: int, optional
                patience for early stopping. Default is np.inf, that is no early stopping.
            - verbose: int, optional
                set after how many epochs the information on losses are printed. Default is 1.
        OUTPUTS:
            - None.
        '''
        self.set_params(params) #Set the parameters
        self.dev = dev
        self.verbose = verbose
        super().__init__()
        # Define the model and optimizer
        self.mdl = FFN(self.params['layers'], self.params['initializer'],
                       self.params['activation'], self.params['dropout']).to(self.dev)
        self.set_optimizer() #Define the optimizer
        # Define the loss function
        self.loss = PinballLoss_MultiQ(
            np.linspace(0, self.params['theta'], self.params['n_points']+1, endpoint=True)[1:])
    
    def set_params(self, params):
        '''
        Define the ultimate parameters dictionary by merging the parameters
            defined by the user with the default ones
        INPUT:
            - params: dict
                parameters defined by the user.
        OUTPUT:
            - None.
        '''
        self.params = {'optimizer': 'Adam', 'reg_type': None,
                       'reg': 0, 'initializer': 'glorot_normal', 'activation': 'relu',
                       'lr': 0.01, 'dropout': 0, 'batch_size':-1,
                       'patience': np.inf, 'verbose': 1} #Default parameters
        self.params.update(params) #Update default parameters with those provided by the user
    
    def __call__(self, x_test):
        '''
        Predict the quantile forecast and the expected shortfall.
        INPUT:
            - x_test: torch.Tensor
                input of the model; shape (batch_size, n_features).
        OUTPUT:
            - qf: ndarray
             quantile forecast of the model.
            - ef: ndarray
                expected shortfall predicted by the model.
        '''
        res = self.mdl(x_test)
        return {'qf':res[:,-1].cpu().detach().numpy(),
                'ef':torch.mean(res, dim=1).cpu().detach().numpy()}
    