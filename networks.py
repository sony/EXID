import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, dropout_prob=0.5):
        super(DDQN, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.head_1 = nn.Linear(self.input_shape[0], layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.ff_2 = nn.Linear(layer_size, action_size)

    def forward(self, input, training=True):
        """
        
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))

        # Apply dropout only during uncertainity calculations
        if not training:
            x = self.dropout(x)
        
        out = self.ff_2(x)
        
        return out

'---------------------------Actor networks for environments------------------------'
class ANetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, dropout_prob=0.5):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(ANetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state, training=True):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        # Apply dropout only during uncertainity calculations
        if not training:
            x = self.dropout(x)
        x = F.relu(x)
        return F.softmax(self.fc3(x))

class MActor(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(MActor, self).__init__()
        self.activation_fc = activation_fc
        self.dropout = nn.Dropout(p=0.5)
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state, training=True):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        if not training:
            x = self.dropout(x)
        x = self.output_layer(x)
        return F.softmax(x)


'''------------------------Networks used in baseline------------------------------'''
class Critic(nn.Module):

    def __init__(self, num_state, num_actions, seed, n_estimates=1):
        super(Critic, self).__init__()

        # set seed
        torch.manual_seed(seed)
        self.num_actions = num_actions
        self.dropout = nn.Dropout(p=0.5)
        num_hidden = 256

        self.backbone = nn.Sequential(
            nn.Linear(in_features=num_state, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU()
        )

        self.out = nn.Linear(in_features=num_hidden, out_features=num_actions * n_estimates)

        for param in self.parameters():
            if len(param.shape) == 1:
                torch.nn.init.constant_(param, 0)
            if len(param.shape) >= 2:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, state, training=True):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0)

        state = self.backbone(state)

        if not training:
            state = self.dropout(state)

        return self.out(state)

class RemCritic(Critic):

    def __init__(self, num_state, num_actions, seed, heads):
        super(RemCritic, self).__init__(num_state, num_actions, seed, heads)

        self.heads = heads

    def forward(self, state):
        state = super(RemCritic, self).forward(state)

        if self.training:
            alphas = torch.rand(self.heads).to(device=state.device)
            alphas /= torch.sum(alphas)

            return torch.sum(state.view(len(state), self.heads, self.num_actions) * alphas.view(1, -1, 1), dim=1)
        else:
            return torch.mean(state.view(len(state), self.heads, self.num_actions), dim=1)


class QrCritic(Critic):

    def __init__(self, num_state, num_actions, seed, quantiles):
        super(QrCritic, self).__init__(num_state, num_actions, seed, quantiles)

        self.quantiles = quantiles

    def forward(self, state):
        state = super(QrCritic, self).forward(state)

        state = state.reshape(len(state), self.num_actions, self.quantiles)

        if self.training:
            return state
        return torch.mean(state, dim=2)

"======================Actor for baseline====================="
class Actor(nn.Module):

    def __init__(self, num_state, num_actions, seed):
        super(Actor, self).__init__()

        # set seed
        torch.manual_seed(seed)

        num_hidden = 256

        self.fnn = nn.Sequential(
            nn.Linear(in_features=num_state, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_actions)
        )

        for param in self.parameters():
            if len(param.shape) == 1:
                torch.nn.init.constant_(param, 0)
            if len(param.shape) >= 2:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0)

        return self.fnn(state)