from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
import torch
import torch.nn.functional as F


class GraphEnsemble(torch.nn.Module):
    """Class for Graph Ensemble"""
    def __init__(self, numInputFeatures: int, numHiddenLayerFeatures: int, numClasses: int, usesWeightedEnsemble = True, ensembleModels: list = None, finalModel = None):
        """Constructor for Graph Ensemble"""
        super().__init__()
        # Input, Output, K, Cached
        
        if (ensembleModels is None):
            # Reserved for whatever is found to be the best
            self.models = torch.nn.ModuleList([GCNConv(numInputFeatures, numHiddenLayerFeatures),
                SAGEConv(numInputFeatures, numHiddenLayerFeatures),
                GATConv(numInputFeatures, numHiddenLayerFeatures)])
        else:
            self.models = torch.nn.ModuleList(ensembleModels)

        self.usesWeightedEnsemble = usesWeightedEnsemble
        self.w = torch.nn.Parameter(torch.randn(len(self.models), requires_grad=True))

        if (finalModel is None):
            self.convFinal = TransformerConv(numHiddenLayerFeatures if len(self.models) > 0 else numInputFeatures, numClasses)
        else:
            self.convFinal = finalModel

    def forward(self, data):
        """Forward pass for the Graph Ensemble"""
        x, edge_index = data.x, data.edge_index

        if (len(self.models) >= 1):
            ensembleOut = F.gelu(self.models[0](x, edge_index))*(self.w[0] if self.usesWeightedEnsemble else 1)

            for i in range(1, len(self.models)):
                ensembleOut += F.gelu(self.models[i](x, edge_index))*(self.w[i] if self.usesWeightedEnsemble else 1)
        else:
            ensembleOut = x


        x = self.convFinal(ensembleOut, edge_index)

        return F.log_softmax(x, dim=1)
    

if (__name__ == "__main__"):
    # Run experiment on PubMed
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid('Datasets','PubMed')

    # Get information about the data set
    numberClasses = max(dataset.y).item() + 1
    numNodes = dataset.data.x.shape[0]
    numFeatures = dataset.data.x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = GraphEnsemble(numFeatures, 5000, numberClasses).to(device), dataset.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # Set Masks
    testRatio = .2
    numTestNodes = int(testRatio * numNodes)

    data.train_mask = torch.zeros(numNodes, dtype=torch.bool)
    data.test_mask  = torch.zeros(numNodes, dtype=torch.bool)
    data.test_mask[:numTestNodes] = True
    data.train_mask[numTestNodes:] = True

    # Train
    for i in range(300):
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
        acc = torch.sum(model(data)[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).item() / data.test_mask.sum().item()
        print(acc)
        loss.backward()
        optimizer.step()