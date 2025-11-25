import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import Planetoid
import torch_geometric.utils as pyg_utils
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric.transforms as T # Import per LCC

# ----------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------
def get_laplacians(edge_index, num_nodes):
    edge_index_L, edge_weight_L = pyg_utils.get_laplacian(
        edge_index, num_nodes=num_nodes, normalization=None
    )
    L = pyg_utils.to_dense_adj(edge_index_L, edge_attr=edge_weight_L)[0]
    edge_index_L_sym, edge_weight_L_sym = pyg_utils.get_laplacian(
        edge_index, num_nodes=num_nodes, normalization='sym'
    )
    L_sym = pyg_utils.to_dense_adj(edge_index_L_sym, edge_attr=edge_weight_L_sym)[0]
    return L, L_sym

def calculate_dirichlet_energy(L: torch.Tensor, x: torch.Tensor) -> float:
    if x.shape[1] == 0: return 0.0
    energy = torch.trace(x.t() @ L @ x) / x.size(0)
    return energy.item()

# ----------------------------------------------------------------------------
# Model Definition
# ---MODIFICATIONS FOR SELF-LOOPS ---
# ----------------------------------------------------------------------------

class PropagationLayer(MessagePassing):
    """
    This layer calculates H(k+1) = A_norm @ H(k)
    """
    def __init__(self):
        super(PropagationLayer, self).__init__(aggr='add')

    def forward(self, x, edge_index, use_self_loops: bool):
        # Calculate the A_norm operator passing the parameter
        edge_index_norm, edge_weight_norm = gcn_norm(
            edge_index, 
            num_nodes=x.size(0), 
            add_self_loops=use_self_loops, # <-- CONTROL HERE
            dtype=x.dtype
        )
        
        # Propagate
        return self.propagate(
            edge_index_norm, x=x, edge_weight=edge_weight_norm
        )

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j

class CustomSequentialModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_layers=24, use_self_loops=True):
        super(CustomSequentialModel, self).__init__()
        
        self.use_self_loops = use_self_loops # Memorize the parameter
        self.first_layer = torch.nn.Linear(num_features, hidden_channels, bias=False)
        self.propagation_layers = torch.nn.ModuleList(
            [PropagationLayer() for _ in range(num_layers - 1)]
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        intermediate_embeddings = [x]
        
        x = self.first_layer(x)
        intermediate_embeddings.append(x)

        # H(k+1) = A_norm @ H(k)
        for i, layer in enumerate(self.propagation_layers):
            # Pass the parameter to the layer
            x = layer(x, edge_index, use_self_loops=self.use_self_loops)
            
            norm = torch.linalg.norm(x, 'fro').item()
            if i % 4 == 0 or i == len(self.propagation_layers) - 1:
                print(f"  ... Model: Layer {i+2}, Norm: {norm:.4e}", flush=True)
            
            intermediate_embeddings.append(x)
            
        return intermediate_embeddings

# ----------------------------------------------------------------------------
# Experiment Execution
# ----------------------------------------------------------------------------

def run_experiment(use_self_loops: bool):
    loops_str = "WithLoops" if use_self_loops else "WithoutLoops"
    print(f"\n--- Experiment: Custom Model on Cora (LCC) | {loops_str} ---", flush=True)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    transform = T.Compose([T.ToUndirected(), T.LargestConnectedComponents()])
    data_lcc = transform(data)
    print(f"Statistics LCC: {data_lcc.num_nodes} nodes, {data_lcc.num_edges} edges", flush=True)
    
    model = CustomSequentialModel(
        num_features=dataset.num_features,
        hidden_channels=16,
        num_layers=50,
        use_self_loops=use_self_loops 
    )
    
    print("Starting Laplacian calculation...", flush=True)
    L, L_sym = get_laplacians(data_lcc.edge_index, data_lcc.num_nodes)
    
    print("Starting model execution...", flush=True)
    with torch.no_grad():
        embeddings_per_layer = model(data_lcc)
    print("Model execution completed.", flush=True)
        
    unnormalized_energies, normalized_energies, frobenius_norms = [], [], []
    for x_layer in embeddings_per_layer:
        unnormalized_energies.append(calculate_dirichlet_energy(L, x_layer))
        normalized_energies.append(calculate_dirichlet_energy(L_sym, x_layer))
        frobenius_norms.append(torch.linalg.norm(x_layer, 'fro').item())

    print("Creating plot...", flush=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE = 24, 22, 20

    ax.plot(unnormalized_energies, marker='o', linestyle='--', label=r'Unnormalized Energy ($E_\Delta / N$)')
    ax.plot(normalized_energies, marker='x', linestyle='-', label=r'Normalized Energy ($E_{\Delta_{norm}} / N$)')
    ax.plot(frobenius_norms, marker='s', linestyle=':', label=r'Frobenius Norm ($\|X\|_F$)', color='green')
    
    title = f"Dirichlet Energies & Norm - Cora LCC ({loops_str})"
    ax.set_title(title, fontsize=TITLE_SIZE)
    
    ax.set_ylabel("Value", fontsize=LABEL_SIZE)
    ax.set_xlabel("Number of Layers", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, which="both", ls="--")
    ax.set_yscale('symlog', linthresh=1e-5)
  
    plt.tight_layout()
    
    output_filename = f"custom_sequential_cora_LCC_{loops_str.lower()}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot saved as '{output_filename}'", flush=True)
    plt.close(fig)

if __name__ == "__main__":
    run_experiment(use_self_loops=False)
