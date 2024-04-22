
import torch
import cupy as cp
import numpy as np
from copy import deepcopy
from itertools import chain
from typing import Optional, Tuple


class AffineSoftplus(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(dim_in, dim_out, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.empty(1, dim_out, dtype=torch.float32))
        self.activation = torch.nn.Softplus()
        self.diff_activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.W) + self.b
        y = self.activation(z)
        return y
    
    @torch.jit.export
    def forward_diff(self, x: torch.Tensor, dy_prev: Optional[torch.Tensor], only_first_diff: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.matmul(x, self.W) + self.b
        diff_activation = self.diff_activation(z).unsqueeze(1)
        if only_first_diff:
            W = self.W[0].unsqueeze(0)
        else:
            W = self.W
        W = W.unsqueeze(0)
        y = self.activation(z)
        if dy_prev is not None:
            dy = (dy_prev @ W) * diff_activation
        else:
            dy = W * diff_activation
        return y, dy

class Affine(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(dim_in, dim_out, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.empty(1, dim_out, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(x, self.W) + self.b
        return y
    
    @torch.jit.export
    def forward_diff(self, x: torch.Tensor, dy_prev: Optional[torch.Tensor], only_first_diff: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.matmul(x, self.W) + self.b
        if only_first_diff:
            W = self.W[0].unsqueeze(0)
        else:
            W = self.W
        W = W.unsqueeze(0)
        if dy_prev is not None:
            dy = dy_prev @ W
        else:
            dy = W
        return y, dy

class ModelRandomAlpha(torch.nn.Module):
    def __init__(self, input_dim: int, num_hidden_layers: int, num_hidden_units: int):
        super().__init__()
        h = []
        dim_in = input_dim
        for i in range(num_hidden_layers):
            h.append(AffineSoftplus(dim_in, num_hidden_units))
            dim_in = num_hidden_units
        self.h = torch.nn.ModuleList(h)
        self.o = Affine(num_hidden_units, 1)
        self.register_buffer('x_mean', torch.zeros(1, input_dim, dtype=torch.float32))
        self.register_buffer('x_std', torch.ones(1, input_dim, dtype=torch.float32))
        self.register_buffer('y_mean', torch.zeros(1, 1, dtype=torch.float32))
        self.register_buffer('y_std', torch.ones(1, 1, dtype=torch.float32))
        for l in chain(self.h, (self.o,)):
            torch.nn.init.normal_(l.W, mean=0., std=np.sqrt(1/l.W.shape[0]))
            torch.nn.init.zeros_(l.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = (x-self.x_mean)/self.x_std
        for l in self.h:
            a = l(a)
        return self.o(a)*self.y_std+self.y_mean
    
    @torch.jit.export
    def forward_diff(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = (x-self.x_mean)/self.x_std
        a, da = self.h[0].forward_diff(a, None, True)
        da = da / self.x_std[:, 0, None, None]
        for l in self.h[1:]:
            a, da = l.forward_diff(a, da, False)
        a, da = self.o.forward_diff(a, da, False)
        a = a * self.y_std + self.y_mean
        da = da[:, :, 0] * self.y_std
        return a, da

class ModelRandomAlphaPiecewiseAffine(torch.nn.Module):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, interpolation_nodes, bottleneck=False, positive_increments=True):
        super().__init__()
        h = []
        dim_in = input_dim-1
        for i in range(num_hidden_layers):
            if bottleneck and i==num_hidden_layers-1:
                h.append(AffineSoftplus(num_hidden_units, 1))
                break
            h.append(AffineSoftplus(dim_in, num_hidden_units))
            dim_in = num_hidden_units
        self.h = torch.nn.ModuleList(h)
        self.o = Affine(num_hidden_units if not bottleneck else 1, interpolation_nodes.shape[0])
        self.positive_increments = positive_increments
        self.register_buffer('x_mean', torch.zeros(1, input_dim, dtype=torch.float32))
        self.register_buffer('x_std', torch.ones(1, input_dim, dtype=torch.float32))
        self.register_buffer('y_mean', torch.zeros(1, 1, dtype=torch.float32))
        self.register_buffer('y_std', torch.ones(1, 1, dtype=torch.float32))
        self.register_buffer('interpolation_nodes', interpolation_nodes)
        self.register_buffer('interpolation_nodes_delta', interpolation_nodes[1:]-interpolation_nodes[:-1])
        for l in chain(self.h, (self.o,)):
            torch.nn.init.normal_(l.W, mean=0., std=np.sqrt(1/l.W.shape[0]))
            torch.nn.init.zeros_(l.b)

    def forward(self, x):
        a = (x[:, 1:]-self.x_mean[:, 1:])/self.x_std[:, 1:]
        for l in self.h:
            a = l(a)
        a = self.o(a)
        if self.positive_increments:
            a = a[:, 0, None]+torch.nn.functional.softplus((torch.minimum(x[:, 0, None], self.interpolation_nodes[None, 1:])-self.interpolation_nodes[None, :-1])*(self.interpolation_nodes[None, :-1]>=x[:, 0, None])*a[:, 1:]/self.interpolation_nodes_delta[None, :]).sum(1, keepdim=True)
        else:
            a = a[:, 0, None]+((torch.minimum(x[:, 0, None], self.interpolation_nodes[None, 1:])-self.interpolation_nodes[None, :-1])*(self.interpolation_nodes[None, :-1]>=x[:, 0, None])*a[:, 1:]/self.interpolation_nodes_delta[None, :]).sum(1, keepdim=True)
        return a*self.y_std+self.y_mean

@torch.jit.script
def _var_loss(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.relu(y_true-y_pred)+alpha*y_pred)

def train_single_var(model, X, Y, X_valid, Y_valid, alpha, batch_size, lr, num_epochs):
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1
    assert isinstance(alpha, torch.Tensor)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters())
    model.x_mean.copy_(X.mean(0)[None, :])
    model.x_std.copy_(X.std(0)[None, :])
    model.y_mean.copy_(Y.mean(0)[None, :])
    model.y_std.copy_(Y.std(0)[None, :])
    with torch.no_grad():
        valid_loss = 0
        for b in range((X_valid.shape[0]+batch_size-1)//batch_size):
            idx = slice(b*batch_size, (b+1)*batch_size)
            Y_pred_valid = model(X_valid[idx])
            valid_loss += _var_loss(Y_pred_valid, Y_valid[idx], alpha)/model.y_std
        valid_loss /= b+1
        valid_loss = valid_loss.item()
    best_state_dict = deepcopy(model.state_dict())
    best_valid_loss = valid_loss
    losses = [(valid_loss, best_valid_loss)]
    for e in range(num_epochs):
        for group in optimizer.param_groups:
            group['lr'] = lr/np.sqrt(e+1)
        for b in range((X.shape[0]+batch_size-1)//batch_size):
            idx = slice(b*batch_size, (b+1)*batch_size)
            optimizer.zero_grad()
            Y_pred = model(X[idx])
            loss = _var_loss(Y_pred, Y[idx], alpha)/model.y_std[0, 0]
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            valid_loss = 0
            for b in range((X_valid.shape[0]+batch_size-1)//batch_size):
                idx = slice(b*batch_size, (b+1)*batch_size)
                Y_pred_valid = model(X_valid[idx])
                valid_loss += _var_loss(Y_pred_valid, Y_valid[idx], alpha)/model.y_std
            valid_loss /= b+1
            valid_loss = valid_loss.item()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            for k, v in best_state_dict.items():
                v.copy_(model.state_dict()[k])
        losses.append((valid_loss, best_valid_loss))
    model.load_state_dict(best_state_dict)
    return losses

def train_single_es_regr(model, X, Y, Y_var, alpha, batch_size):
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1
    assert isinstance(alpha, torch.Tensor)
    device = next(model.parameters()).device
    model.x_mean.copy_(X.mean(0)[None, :])
    model.x_std.copy_(X.std(0)[None, :])
    with torch.no_grad():
        es_Y = (Y-Y_var).relu_().div_(alpha)+Y_var
    model.y_mean.copy_(es_Y.mean(0)[None, :])
    model.y_std.copy_(es_Y.std(0)[None, :])
    h_aug = torch.empty((batch_size, model.o.W.shape[0]+1), dtype=torch.float32, device=device)
    h_aug[:, 0] = 1
    with torch.no_grad():
        model.o.W.zero_()
        model.o.b.zero_()
        for b in range((X.shape[0]+batch_size-1)//batch_size):
            idx = slice(b*batch_size, (b+1)*batch_size)
            h = X[idx]
            for l in model.h:
                h = l(h)
            h_aug[:, 1:] = h
            with cp.cuda.Device(device.index):
                sol, _, _, _ = cp.linalg.lstsq(cp.asarray(h_aug), cp.asarray((es_Y[idx]-model.y_mean)/model.y_std), rcond=None)
                sol = torch.as_tensor(sol, device=device)
            model.o.W.add_(sol[1:h_aug.shape[1]])
            model.o.b.add_(sol[:1])
        model.o.W /= b+1
        model.o.b /= b+1

class BCGNS():
    def __init__(self, theta, dim, device):
        # BCGNS model computes VaR and ES as the positive tail of the loss. Adjust accordingly
        self.theta = torch.tensor(1-theta, device=device)
        # Construct the models
        self.var_mdl = torch.jit.script(ModelRandomAlpha(dim, 3, 2*dim)).to(device)
        self.es_mdl = torch.jit.script(ModelRandomAlpha(dim, 3, 2*dim)).to(device)
    
    def fit(self, x_train_, y_train_, x_val_, y_val_, batch_size):
        # BCGNS model computes VaR and ES as the positive tail of the loss. Adjust accordingly
        x_train, y_train, x_val, y_val = -x_train_, -y_train_, -x_val_, -y_val_
        # Train the var model
        _ = train_single_var(self.var_mdl, x_train, y_train,
                             x_val, y_val, self.theta, batch_size, 0.01, 2000)
        # Train the es model
        x_tv = torch.cat([x_train, x_val], dim=0)
        x_tv = x_tv[x_tv.shape[0] % batch_size:]
        y_tv = torch.cat([y_train, y_val], dim=0)
        y_tv = y_tv[y_tv.shape[0] % batch_size:]
        self.es_mdl.load_state_dict(self.var_mdl.state_dict())
        with torch.no_grad():
            y_es_train = self.var_mdl(x_tv)
        train_single_es_regr(
            self.es_mdl, x_tv, y_tv, y_es_train, self.theta, batch_size)
    
    def __call__(self, x_test_):
        # BCGNS model computes VaR and ES as the positive tail of the loss. Adjust accordingly
        x_test = -x_test_
        var = self.var_mdl(x_test).cpu().detach().numpy()
        r = self.es_mdl(x_test).cpu().detach().numpy()
        return {'qf':var, 'ef':r*(1-self.theta.item()) + var}