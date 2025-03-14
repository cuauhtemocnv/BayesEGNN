from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.mixture import GaussianMixture

def initialize_weights_egnn(model):
    """
    Initializes the weights of the given EGNN model using Xavier initialization.
    
    Args:
    - model (nn.Module): The neural network model to initialize.
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            init.xavier_normal_(param)  # Xavier initialization (normal distribution)
        elif 'bias' in name:
            init.zeros_(param)  # Bias initialized to zero

# Example usage for your EGNN model
class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_layers):
        super(EGNN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_node_nf, hidden_nf)] + 
                                    [nn.Linear(hidden_nf, hidden_nf) for _ in range(n_layers-1)] +
                                    [nn.Linear(hidden_nf, out_node_nf)])
        self.n_layers = n_layers

    def forward(self, node_features, node_coords, edge_indices, edge_features):
        x = node_features
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x, node_coords  # For simplicity, return node coordinates unchanged here
    
class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=True, coords_agg='mean', tanh=True):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=3, residual=False, attention=False, normalize=True, tanh=False):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x)
        h = self.embedding_out(h)
        return h, x

    def get_hidden_representation(self, h, x, edges, edge_attr=None, layer_index=-1):
        """
        Extract hidden representation from a specific layer.
        
        Args:
            h (torch.Tensor): Input node features.
            x (torch.Tensor): Node coordinates.
            edges (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor): Edge attributes.
            layer_index (int): The index of the layer to extract features from (default: -1 for the last layer).
        
        Returns:
            torch.Tensor: The hidden representation from the specified layer.
        """
        # Apply initial embedding
        h = self.embedding_in(h)
        
        # Iterate through layers to extract features
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x)
            if i == layer_index:  # Stop at the desired layer
                return h
        
        # If layer_index is -1, return output embedding
        return self.embedding_out(h)


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def get_edges(points, cutoff):
    """
    Creates edge index and edge distances for a point cloud within a cutoff radius.

    Args:
        points: A PyTorch tensor of shape [num_points, 3] representing the point cloud.
        cutoff: The cutoff radius.

    Returns:
        edge_index: A PyTorch tensor of shape [2, num_edges] representing the edge index.
        edge_attr: A PyTorch tensor of shape [num_edges, 1] representing the edge distances.
    """

    dist_mat = torch.cdist(points, points)
    mask = dist_mat > cutoff
    dist_mat[mask] = 0

    edge_index = torch.nonzero(dist_mat, as_tuple=True)
    edge_attr = dist_mat[edge_index[0], edge_index[1]].unsqueeze(1)

    return edge_index, edge_attr    
# GMM Training and Reliability Check
def train_egnn_gmm(egnn, data_loader, n_epochs=100, gmm_epochs=20, n_components=3):
    optimizer = Adam(egnn.parameters(), lr=0.001)
    gmm = GaussianMixture(n_components=n_components)

    for epoch in range(1, n_epochs + 1):
        for data in data_loader:  # Assume data_loader provides graph data
            node_features, edge_index, targets = data
            optimizer.zero_grad()
            predictions = egnn(node_features, edge_index)
            loss = nn.MSELoss()(predictions, targets)
            loss.backward()
            optimizer.step()
        
        # Train GMM every `gmm_epochs`
        if epoch % gmm_epochs == 0:
            hidden_representations = []
            with torch.no_grad():
                for data in data_loader:
                    node_features, edge_index, _ = data
                    hidden = egnn.get_hidden_representation(node_features, edge_index)
                    hidden_representations.append(hidden.cpu().numpy())
            hidden_representations = np.vstack(hidden_representations)
            gmm.fit(hidden_representations)
            print(f"Epoch {epoch}: GMM trained on latent space.")

    return egnn, gmm

# Reliability Check
def compute_reliability(egnn, gmm, new_data):
    node_features, edge_index = new_data
    with torch.no_grad():
        hidden = egnn.get_hidden_representation(node_features, edge_index)
    nll = -gmm.score_samples(hidden.cpu().numpy())  # Negative log-likelihood
    reliability = np.mean(nll)
    print(f"Negative Log-Likelihood (NLL): {reliability:.4f}")
    return reliability
