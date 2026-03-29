import torch.nn.functional as F
import torch.nn as nn

def make_mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim)
    )

class VesselGNN(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, num_classes, n_layers=4, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import GINEConv
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # encode edge features → hidden size
        self.edge_encoder = nn.Linear(edge_dim, hidden_channels)

        # first layer
        self.convs.append(
            GINEConv(make_mlp(in_channels, hidden_channels), edge_dim=hidden_channels)
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # hidden layers
        for _ in range(n_layers - 1):
            self.convs.append(
                GINEConv(make_mlp(hidden_channels, hidden_channels), edge_dim=hidden_channels)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout    = dropout

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)

        # first layer — no residual (different dims)
        x = F.relu(self.bns[0](self.convs[0](x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # remaining layers — with residual
        for conv, bn in zip(self.convs[1:], self.bns[1:]):
            x = x + F.dropout(F.relu(bn(conv(x, edge_index, edge_attr))), 
                            p=self.dropout, training=self.training)

        return self.classifier(x)
    
class VesselSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, n_layers=4, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout    = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)