from torch import nn
import torch
import math
import numpy as np
import logging
#import pdb

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Sequential

import torch
from torch import nn


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class GCLPyG(MessagePassing):
    def __init__(self, input_nf, output_nf, hidden_nf, act_fn=nn.ReLU(), attention=False):
        super(GCLPyG, self).__init__(aggr='add')  # Use 'add' for summation aggregation.

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        # Attention, if necessary
        self.attention = attention
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )

    def forward(self, x, edge_index, edge_attr=None):
        # Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Message computation
        edge_features = torch.cat([x_i, x_j], dim=1)
        edge_features = self.edge_mlp(edge_features)
        if self.attention:
            att_val = self.att_mlp(edge_features)
            edge_features = edge_features * att_val
        return edge_features

    def update(self, aggr_out, x):
        # Node update
        new_features = torch.cat([x, aggr_out], dim=1)
        return self.node_mlp(new_features)


class EquivariantUpdatePyG(MessagePassing):
    def __init__(self, hidden_nf, edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdatePyG, self).__init__(aggr='add')  # Default aggregation method is 'add'
        self.tanh = tanh
        self.coords_range = coords_range

        # Coordinate MLP
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, edge_mask=None):
        # Propagate the messages
        return self.propagate(edge_index, size=(coord.size(0), coord.size(0)), 
                              h=h, coord=coord, coord_diff=coord_diff, 
                              edge_attr=edge_attr, edge_mask=edge_mask)

    def message(self, h_i, h_j, coord_i, coord_diff, edge_attr, edge_mask):
        # Message computation
        input_tensor = torch.cat([h_i, h_j, edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        
        if edge_mask is not None:
            trans = trans * edge_mask
        return trans

    def update(self, aggr_out, coord):
        # Update coordinates based on aggregated messages
        return coord + aggr_out

class EquivariantBlockPyG(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, act_fn=nn.SiLU(), n_layers=2, attention=True,
                 tanh=False, coords_range=15, norm_constant=1, sin_embedding=None):
        super(EquivariantBlockPyG, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding

        # Define layers
        layers = []
        for i in range(n_layers):
            layers.append((GCLPyG(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                  act_fn=act_fn, attention=attention),
                          'x, edge_index -> x'))
        layers.append((EquivariantUpdatePyG(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                           coords_range=self.coords_range_layer),
                      'x, coord, edge_index -> coord'))

        self.layers = Sequential('x, coord, edge_index', layers)

    def forward(self, h, x, edge_index, edge_attr=None, node_mask=None, edge_mask=None):
        # Prepare edge features
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1) if edge_attr is not None else distances

        # Pass through layers
        for layer in self.layers.children():
            if isinstance(layer, EquivariantUpdatePyG):
                x = layer(h, x, edge_index, coord_diff, edge_attr, edge_mask)
            else:
                h = layer(h, edge_index, edge_attr)

        # Apply node mask if provided
        if node_mask is not None:
            h = h * node_mask

        return h, x


class Clof_GCLPyG(MessagePassing):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), 
                 recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False,
                 coords_range=1, norm_constant=0, out_basis_dim=3, clamp=False, 
                 normalization_factor=1,aggregation_method='add'):  # Changed 'sum' to 'add' for PyG
        super(Clof_GCLPyG, self).__init__(aggr=aggregation_method)
        
        # Edge MLP
        input_edge = input_nf * 2 + 1 + edges_in_d  # +1 for radial dimension
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        
        self.layer_norm = nn.LayerNorm(hidden_nf)

        # Coordinate MLP
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, out_basis_dim)]
        if tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        # Attention, if used
        self.attention = attention
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        # Other attributes
        self.recurrent = recurrent
        self.coords_weight = coords_weight
        self.clamp = clamp

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        # Coordinate transformation
        radial, coord_diff, coord_cross, coord_vertical = self.coord2localframe(edge_index, coord)

        # Propagate messages
        h, edge_feat = self.propagate(edge_index, x=h, coord=coord, radial=radial, 
                                      coord_diff=coord_diff, coord_cross=coord_cross, 
                                      coord_vertical=coord_vertical, edge_attr=edge_attr, 
                                      node_attr=node_attr, edge_mask=edge_mask)

        # Coordinate update
        coord = self.coord_model(coord, edge_index, coord_diff, coord_cross, coord_vertical, edge_feat)

        # Apply layer normalization
        h = self.layer_norm(h)

        # Apply node mask if provided
        if node_mask is not None:
            h = h * node_mask

        return h, coord, edge_attr

    def message(self, x_i, x_j, radial_i, coord_diff_i, coord_cross_i, coord_vertical_i, edge_attr, edge_mask):
        # Edge feature computation
        edge_features = torch.cat([x_i, x_j, radial_i, edge_attr], dim=1) if edge_attr is not None else torch.cat([x_i, x_j, radial_i], dim=1)
        edge_features = self.edge_mlp(edge_features)

        # Apply attention if used
        if self.attention:
            att_val = self.att_mlp(edge_features)
            edge_features = edge_features * att_val

        # Apply edge mask if provided
        if edge_mask is not None:
            edge_features = edge_features * edge_mask

        return edge_features

    def update(self, aggr_out, x, node_attr):
        # Node feature update
        agg = torch.cat([x, aggr_out, node_attr], dim=1) if node_attr is not None else torch.cat([x, aggr_out], dim=1)
        out = self.node_mlp(agg)

        # Apply recurrent connection if used
        if self.recurrent:
            out = x + out

        return out
    
    def coord2localframe(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])

        epsilon = 1e-8
        if self.norm_diff:
            norm = torch.sqrt(radial + epsilon) + 1
            coord_diff = coord_diff / norm
            cross_norm = (torch.sqrt(torch.sum((coord_cross)**2, 1).unsqueeze(1) + epsilon)) + 1
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross)
        return radial, coord_diff, coord_cross, coord_vertical

    def coord_model(self, coord, edge_index, coord_diff, coord_cross, coord_vertical, edge_feat):
        row, col = edge_index
        coff = self.coord_mlp(edge_feat)
        trans = coord_diff * coff[:, :1] + coord_cross * coff[:, 1:2] + coord_vertical * coff[:, 2:3]
        if self.clamp:
            trans = torch.clamp(trans, min=-100, max=100)
        agg = self.aggregate(trans, edge_index=row, dim_size=coord.size(0))
        coord += agg * self.coords_weight
        return coord

class EGNNPyG(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='add'):
        super(EGNNPyG, self).__init__()

        if out_node_nf is None:
            out_node_nf = in_node_nf

        self.hidden_nf = hidden_nf
        self.coords_range_layer = float(coords_range/n_layers) if n_layers > 0 else float(coords_range)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = in_edge_nf

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        self.layers = nn.ModuleList([
            EquivariantBlockPyG(hidden_nf, edge_feat_nf=edge_feat_nf, act_fn=act_fn, n_layers=inv_sublayers,
                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                               coords_range=coords_range, norm_constant=norm_constant,
                               sin_embedding=self.sin_embedding,
                               normalization_factor=self.normalization_factor,
                               aggregation_method=aggregation_method)
            for _ in range(n_layers)
        ])

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        # Calculate distances if needed
        distances = None
        if self.sin_embedding:
            distances = self.sin_embedding(coord2diff(x, edge_index)[0])

        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)

        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x

class ClofNetPyG(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, out_node_nf=None, 
                 act_fn=nn.SiLU(), n_layers=4, attention=False, norm_constant=1,
                 coords_weight=1.0, recurrent=False, norm_diff=True, tanh=False):
        super(ClofNetPyG, self).__init__()
        
        if out_node_nf is None:
            out_node_nf = in_node_nf

        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.norm_diff = norm_diff

        # Node feature embedding layers
        self.embedding_node_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_node_out = nn.Linear(self.hidden_nf, out_node_nf)

        # Edge feature processing
        edge_embed_dim = 8  # Adjust as needed
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
            nn.Linear(self.hidden_nf // 2, self.hidden_nf // 2), act_fn)

        # Graph convolutional layers
        self.layers = nn.ModuleList([
            Clof_GCLPyG(
                input_nf=self.hidden_nf,
                output_nf=self.hidden_nf,
                hidden_nf=self.hidden_nf,
                edges_in_d=self.hidden_nf // 2,
                act_fn=act_fn,
                recurrent=recurrent,
                coords_weight=coords_weight,
                norm_diff=norm_diff,
                tanh=tanh)
            for _ in range(n_layers)
        ])

    def scalarization(self, edge_index, x):
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edge_index, x)

        # Geometric Vectors Scalarization
        row, col = edge_index
        edge_basis = torch.cat([coord_diff, coord_cross, coord_vertical], dim=1)
        
        r_i = x[row]  
        r_j = x[col]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)
        
        # Calculate angle information in local frames
        coff_mul = coff_i * coff_j  # Element-wise multiplication
        coff_i_norm = torch.norm(coff_i, dim=-1, keepdim=True) + 1e-5
        coff_j_norm = torch.norm(coff_j, dim=-1, keepdim=True) + 1e-5
        pseudo_cos = torch.sum(coff_mul, dim=-1, keepdim=True) / (coff_i_norm * coff_j_norm)
        pseudo_sin = torch.sqrt(1 - pseudo_cos**2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)
        
        # Concatenating features
        coff_feat = torch.cat([pseudo_angle, coff_i, coff_j], dim=-1)
        return coff_feat


    def forward(self, h, edge_index, x, n_nodes=5, 
                edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        h = self.embedding_node_in(h)

        # Process node features
        x = x.view(-1, n_nodes, 3)
        centroid = x.mean(dim=1, keepdim=True)
        x_center = (x - centroid).view(-1, 3)

        # Scalarization of edge features
        coff_feat = self.scalarization(edge_index, x_center)
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1) if edge_attr is not None else coff_feat
        edge_feat = self.fuse_edge(edge_feat)

        # Pass through graph convolutional layers
        for layer in self.layers:
            h, x_center, _ = layer(h, edge_index, x_center, edge_attr=edge_feat, 
                                   node_attr=node_attr, node_mask=node_mask, edge_mask=edge_mask)

        # Final transformations
        x = x_center.view(-1, n_nodes, 3) + centroid
        x = x.view(-1, 3)
        h = self.embedding_node_out(h)

        if node_mask is not None:
            h = h * node_mask

        return h, x

class GNNPyG(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='add', 
                 act_fn=nn.SiLU(), n_layers=4, attention=False, 
                 normalization_factor=1, out_node_nf=None):
        super(GNNPyG, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf

        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        # Node feature embedding layers
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        # Graph convolutional layers
        self.layers = nn.ModuleList([
            GCLPyG(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention)
            for _ in range(n_layers)
        ])

    def forward(self, h, edge_index, edge_attr=None, node_mask=None, edge_mask=None):
        h = self.embedding(h)
        for layer in self.layers:
            h, _ = layer(h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        if node_mask is not None:
            h = h * node_mask

        return h
