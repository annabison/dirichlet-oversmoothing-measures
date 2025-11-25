import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import get_laplacian, degree, to_dense_adj
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Function to Calculate Dirichlet Energy ---
def calculate_dirichlet(H, L):
    """
    Calculates the Dirichlet energy Tr(H^T * L * H)
    H: Embedding matrix (Nodes x Features)
    L: Laplacian matrix (Nodes x Nodes)
    """
    try:
        energy = torch.trace(H.t() @ L @ H)
        return energy.item()
    except Exception as e:
        print(f"Error in calculate_dirichlet: {e}")
        return 0.0

# --- 2. Main Experiment Function (for a single graph) ---
def run_experiment_on_graph(data, graph_name, num_layers=50):
    """
    Runs the full experiment on a single graph (data) and saves the energy ratio plots.
    """
    
    # --- 3. Graph Data Preparation ---
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    if data.x is None:
        print(f"Graph {graph_name} has no features, skipping.")
        return
        
    initial_features = data.x.to(torch.float64)
    
    print(f"\n--- Starting Experiment for: {graph_name} ---")
    print(f"Nodes: {num_nodes}, Edges: {edge_index.shape[1]}")

    # --- 4. Matrix Preparation (Manual for Double Precision) ---
    
    # Adjacency Matrix (A)
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].to(torch.float64)
    I = torch.eye(num_nodes, dtype=torch.float64)

    # Calculation of D, D^-0.5
    degrees_vec = degree(edge_index[0], num_nodes=num_nodes).to(torch.float64)
    D_matrix = torch.diag(degrees_vec)
    degrees_for_inv = degrees_vec.clone()
    degrees_for_inv[degrees_for_inv == 0] = 1.0 
    d_inv_sqrt_vec = 1.0 / torch.sqrt(degrees_for_inv)
    D_inv_sqrt_matrix = torch.diag(d_inv_sqrt_vec)

    # A) Standard Laplacian (L_std = D - A)
    L_std = D_matrix - A

    # B) Symmetric Normalized Laplacian (L_sym = I - D^-0.5 * A * D^-0.5)
    L_sym = I - D_inv_sqrt_matrix @ A @ D_inv_sqrt_matrix

    # C) GCN Propagation Matrix (A_norm)
    A_norm =  D_inv_sqrt_matrix @ A @ D_inv_sqrt_matrix
    
    # --- 5. Iteration Execution ---
    layers = []
    energy_ratios = []
    norm_e_std_list = []  
    norm_e_sym_list = []  
    frobenius_norm_list = []

    H = initial_features

    for layer in range(num_layers + 1):
        e_std = calculate_dirichlet(H, L_std)
        e_sym = calculate_dirichlet(H, L_sym)
        f_norm = torch.norm(H, p='fro').item()
        
        norm_e_std = e_std / num_nodes
        norm_e_sym = e_sym / num_nodes
        
        # Calculate the ratio
        ratio = norm_e_std / (norm_e_sym + 1e-12) if norm_e_sym != 0 else 0
        
        layers.append(layer)
        energy_ratios.append(ratio)
        norm_e_std_list.append(norm_e_std) # <-- SAVE ENERGY
        norm_e_sym_list.append(norm_e_sym) # <-- SAVE ENERGY
        frobenius_norm_list.append(f_norm) # <-- SAVE FROBENIUS NORM    
        
        if layer < num_layers:
            H = A_norm @ H
            
    print(f"Completed. Final ratio (E_std/E_norm): {energy_ratios[-1]:.4f}")

    # --- 6. Creation and Saving of Energy Ratio Plot ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(layers, energy_ratios, label=r'Energy Ratio ($E_{std} / E_{norm}$)', marker='d', linestyle='-.')
    

    ax1.set_xlabel('GCN Layer (k)', fontsize=14)
    ax1.set_ylabel('Energy Ratio', fontsize=14)
    ax1.set_title(f'Dirichlet Energy Ratio vs. GCN Layers ({graph_name})', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle=':')
    
    plot_filename_1 = f"energy_ratio_plot_{graph_name}.png"
    fig1.savefig(plot_filename_1)
    print(f"Ratio plot saved in: {os.path.abspath(plot_filename_1)}")
    plt.close(fig1)

    # --- 7. NEW PLOT: Separate Energies ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(layers, norm_e_std_list, label=r'$E_{std}$ (Standard)', marker='o', linestyle='-')
    ax2.plot(layers, norm_e_sym_list, label=r'$E_{norm}$ (Normalized)', marker='x', linestyle='--')
    ax2.plot(layers, frobenius_norm_list, label='Frobenius Norm ||X||_F', color='green', marker='s')

    ax2.set_xlabel('GCN Layer (k)', fontsize=14)
    ax2.set_ylabel('Normalized Dirichlet Energy', fontsize=14)
    ax2.set_title(f'Dirichlet Energies vs. GCN Layers ({graph_name})', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle=':')
    ax2.set_yscale('log') # Logarithmic scale to observe decay

    plot_filename_2 = f"energies_plot_{graph_name}.png"
    fig2.savefig(plot_filename_2)
    print(f"Energies plot saved in: {os.path.abspath(plot_filename_2)}")
    plt.close(fig2)
    
# --- 8. Dataset Loading and Graph Search ---
try:
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
except Exception as e:
    print(f"Error loading ENZYMES dataset: {e}")
    exit()

print(f"Dataset loaded: {dataset.name}. Number of graphs: {len(dataset)}")

found_non_regular = False
found_regular = False

for i, data in enumerate(dataset):
    if data.edge_index is None:
        continue
        
    degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    is_regular = (degrees.min() == degrees.max())
    
    if not found_non_regular and not is_regular:
        run_experiment_on_graph(data, f"ENZYMES_NonRegular_Graph{i}")
        found_non_regular = True
        
    if not found_regular and is_regular: #and not data.num_nodes == 4: 
        run_experiment_on_graph(data, f"ENZYMES_Regular_Graph{i}")
        found_regular = True

    if found_non_regular and found_regular:
        break

# --- 9. Final Messages ---
if not found_non_regular:
    print("\nWARNING: No NON-regular graph was found in the ENZYMES dataset.")
if not found_regular:
    print("\nWARNING: No REGULAR graph was found in the ENZYMES dataset.")
print("\nExecution completed.")