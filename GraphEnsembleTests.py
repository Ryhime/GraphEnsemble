from GraphEnsemble import GraphEnsemble
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv


# Data set info
from torch_geometric.datasets import Planetoid
dataset = Planetoid('Datasets','PubMed')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset.to(device)

numberClasses = max(dataset.y).item() + 1
numNodes = dataset.data.x.shape[0]
numFeatures = dataset.data.x.shape[1]

# File to write results
results = open('ResultsPubMed.csv', 'w')
results.write('Model,MxAccuracy,Accuracy\n')

def runTest(model, numEpochs, title):
    # Get information about the data set
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # Set Masks
    testRatio = .2
    numTestNodes = int(testRatio * numNodes)

    data.train_mask = torch.zeros(numNodes, dtype=torch.bool)
    data.test_mask  = torch.zeros(numNodes, dtype=torch.bool)
    data.test_mask[:numTestNodes] = True
    data.train_mask[numTestNodes:] = True

    # Train
    mxAcc = 0
    for _ in range(numEpochs):
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        acc = torch.sum(model(data)[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).item() / data.test_mask.sum().item()
        if (acc > mxAcc):
            mxAcc = acc

    # Write to file
    results.write(title + ',' + str(mxAcc) + ',' +  str(acc) + '\n')
    print(title + ',' + str(mxAcc) + ',' + str(acc) + '\n')


# Run tests
numEpochs = 500
numHiddenFeatures = 1000

# Classic
m1 = GraphEnsemble(numFeatures, numHiddenFeatures, numberClasses)
runTest(m1, numEpochs, 'GraphEnsemble')

# Classic no weighted
m1 = GraphEnsemble(numFeatures, numHiddenFeatures, numberClasses, usesWeightedEnsemble=False)
runTest(m1, numEpochs, 'GraphEnsembleNoWeighted')

# All Graph Sage Weighted
models = [SAGEConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures)]

m2 = GraphEnsemble(numFeatures, numHiddenFeatures, numberClasses, ensembleModels=models, usesWeightedEnsemble=True)
runTest(m2, numEpochs, 'GraphSage')

# All Graph Sage not weighted
m2 = GraphEnsemble(numFeatures, numHiddenFeatures, numberClasses, ensembleModels=models, usesWeightedEnsemble=False)
runTest(m2, numEpochs, 'GraphSageNoWeighted')

# None
m3 = GraphEnsemble(numFeatures, numHiddenFeatures, numberClasses, ensembleModels=[])
runTest(m3, numEpochs, 'None')

exit()

# GAT
models = [GATConv(numFeatures, numHiddenFeatures),
                GATConv(numFeatures, numHiddenFeatures),
                GATConv(numFeatures, numHiddenFeatures)]

m3 = GraphEnsemble(numFeatures, numHiddenFeatures, numberClasses, ensembleModels=models)
runTest(m3, numEpochs, 'GAT')


# GCN
models = [GCNConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures)]
m4 = GraphEnsemble(numFeatures, numHiddenFeatures, numberClasses, ensembleModels=models)
runTest(m4, 300, 'GCN')

# Lots and lots of layers of GCN
models = [GCNConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures)]

m5 = GraphEnsemble(numFeatures, numHiddenFeatures, numberClasses, ensembleModels=models)
runTest(m5, numEpochs, 'GCN9')


# Lots and lots of layers ensemble
models = [GCNConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures),
                GATConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures),
                GATConv(numFeatures, numHiddenFeatures),
                GCNConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures),
                GATConv(numFeatures, numHiddenFeatures)]

m6 = GraphEnsemble(numFeatures, numHiddenFeatures, numberClasses, ensembleModels=models)
runTest(m6, numEpochs, 'Ensemble9')

# Lots and lots of SAGE
models = [SAGEConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures),
                SAGEConv(numFeatures, numHiddenFeatures)]

m7 = GraphEnsemble(numFeatures, numHiddenFeatures, numberClasses, ensembleModels=models)
runTest(m7, numEpochs, 'SAGE9')
          

results.close()