# esn_gru_3d_dynamics.py
"""
3D Visualization of ESN Reservoir Dynamics and Cultural Context Injection
Shows how cultural information flows through the reservoir network
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.decomposition import PCA

from cultural_context import (
    EnhancedESNCulturalController,
    EnhancedGRUCulturalController
)

# ============================================================
# 🎨 DESIGN SYSTEM
# ============================================================

COLORS = {
    'bg': '#0a0e27',
    'card': '#141b3d',
    'text': '#e8edf4',
    'muted': '#8b93b0',
    'primary': '#3b82f6',
    'secondary': '#8b5cf6',
    'accent': '#f59e0b',
    'success': '#10b981',
    'danger': '#ef4444',
    'grid': '#1e2749',
    'esn': '#3b82f6',
    'gru': '#8b5cf6',
    'cultural': '#f59e0b'
}

# ============================================================
# 1️⃣ 3D ESN RESERVOIR NETWORK WITH CULTURAL FLOW
# ============================================================

def visualize_3d_esn_reservoir_network(
    caption: str,
    esn: EnhancedESNCulturalController,
    steps: int = 10,
    sample_neurons: int = 100,
    output_path: str = "viz_server/static/esn_reservoir_3d.html"
):
    """
    3D visualization of ESN reservoir with neurons as nodes.
    Shows how cultural context flows through the network.
    """
    # Reset and process
    esn.state.fill(0.0)
    x = esn.extract_features(caption)
    
    # Sample neurons for visualization (too many is messy)
    total_neurons = esn.reservoir_size
    if sample_neurons < total_neurons:
        neuron_indices = np.random.choice(total_neurons, sample_neurons, replace=False)
    else:
        neuron_indices = np.arange(total_neurons)
        sample_neurons = total_neurons
    
    # Get sampled weight matrix
    W_sample = esn.W[neuron_indices][:, neuron_indices]
    
    # Create network graph for 3D layout
    G = nx.Graph()
    edges = []
    for i in range(sample_neurons):
        for j in range(i+1, sample_neurons):
            if abs(W_sample[i, j]) > 0.01:  # Only strong connections
                edges.append((i, j, {'weight': abs(W_sample[i, j])}))
    
    G.add_edges_from([(e[0], e[1]) for e in edges])
    
    # Use spring layout in 3D
    pos_2d = nx.spring_layout(G, dim=2, seed=42)
    
    # Create 3D positions (add z-dimension based on activation)
    pos_3d = {}
    
    # Track activations over time
    activation_history = []
    for step in range(steps):
        update = esn.Win @ x + esn.W @ esn.state
        esn.state = (1 - esn.leak_rate) * esn.state + esn.leak_rate * np.tanh(update)
        activation_history.append(esn.state[neuron_indices].copy())
    
    # Use final activation for z-coordinate
    final_activation = activation_history[-1]
    
    for idx, node in enumerate(G.nodes()):
        if node in pos_2d:
            x_coord, y_coord = pos_2d[node]
            z_coord = final_activation[node]  # Height = activation strength
            pos_3d[node] = (x_coord, y_coord, z_coord)
    
    # Create 3D visualization
    fig = go.Figure()
    
    # Add edges (connections between neurons)
    edge_x, edge_y, edge_z = [], [], []
    for edge in edges:
        i, j = edge[0], edge[1]
        if i in pos_3d and j in pos_3d:
            x0, y0, z0 = pos_3d[i]
            x1, y1, z1 = pos_3d[j]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
    
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(59, 130, 246, 0.2)', width=1),
        hoverinfo='none',
        name='Reservoir Connections'
    ))
    
    # Add nodes (neurons) colored by activation
    node_x = [pos_3d[node][0] for node in G.nodes() if node in pos_3d]
    node_y = [pos_3d[node][1] for node in G.nodes() if node in pos_3d]
    node_z = [pos_3d[node][2] for node in G.nodes() if node in pos_3d]
    node_colors = [final_activation[node] for node in G.nodes() if node in pos_3d]
    
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=8,
            color=node_colors,
            colorscale='Viridis',
            colorbar=dict(
                title="Activation<br>Strength",
                thickness=15,
                len=0.7
            ),
            line=dict(color='white', width=0.5)
        ),
        text=[f"Neuron {node}<br>Activation: {final_activation[node]:.3f}" 
              for node in G.nodes() if node in pos_3d],
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Reservoir Neurons'
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b style='font-size:20px'>ESN Reservoir: Cultural Context Injection</b><br>"
                 f"<span style='font-size:14px;color:{COLORS['muted']}'>3D network showing how '{caption[:50]}...' activates the reservoir</span>",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='Network Space X', backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid']),
            yaxis=dict(title='Network Space Y', backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid']),
            zaxis=dict(title='Cultural Activation', backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid']),
            bgcolor=COLORS['card']
        ),
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], family="Inter, system-ui, sans-serif"),
        height=800,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(20, 27, 61, 0.8)',
            bordercolor=COLORS['grid'],
            borderwidth=1
        )
    )
    
    fig.write_html(output_path)
    return output_path


# ============================================================
# 2️⃣ 3D TRAJECTORY: HOW CULTURAL CONTEXT EVOLVES OVER TIME
# ============================================================

def visualize_3d_cultural_trajectory(
    caption: str,
    esn: EnhancedESNCulturalController,
    gru: EnhancedGRUCulturalController,
    steps: int = 20,
    output_path: str = "viz_server/static/cultural_trajectory_3d.html"
):
    """
    3D trajectory showing how cultural understanding evolves over processing steps.
    Compares ESN vs GRU paths through semantic space.
    """
    # ESN processing
    esn.state.fill(0.0)
    x_esn = esn.extract_features(caption)
    esn_states = []
    
    for step in range(steps):
        update = esn.Win @ x_esn + esn.W @ esn.state
        esn.state = (1 - esn.leak_rate) * esn.state + esn.leak_rate * np.tanh(update)
        esn_states.append(esn.state.copy())
    
    esn_states = np.array(esn_states)
    
    # GRU processing (simulate step-by-step)
    import torch
    gru_states = []
    
    x_gru = gru.extract_features(caption)
    hidden = torch.zeros(gru.num_layers, 1, gru.hidden_size)
    
    for step in range(steps):
        with torch.no_grad():
            output, hidden = gru.gru(x_gru, hidden)
            gru_states.append(hidden[-1].squeeze().cpu().numpy())
    
    gru_states = np.array(gru_states)
    
    # Reduce to 3D using separate PCAs (since dimensions differ)
    # ESN: 200D → 3D
    pca_esn = PCA(n_components=3)
    esn_3d = pca_esn.fit_transform(esn_states)
    
    # GRU: 128D → 3D
    pca_gru = PCA(n_components=3)
    gru_3d = pca_gru.fit_transform(gru_states)
    
    # Get explained variance for both
    esn_explained = pca_esn.explained_variance_ratio_
    gru_explained = pca_gru.explained_variance_ratio_
    
    # Create figure
    fig = go.Figure()
    
    # ESN trajectory
    fig.add_trace(go.Scatter3d(
        x=esn_3d[:, 0],
        y=esn_3d[:, 1],
        z=esn_3d[:, 2],
        mode='lines+markers',
        line=dict(color=COLORS['esn'], width=4),
        marker=dict(
            size=6,
            color=np.arange(steps),
            colorscale='Blues',
            showscale=False,
            line=dict(color='white', width=1)
        ),
        name='ESN Path',
        hovertemplate='<b>ESN Step %{marker.color}</b><br>Dim1: %{x:.3f}<br>Dim2: %{y:.3f}<br>Dim3: %{z:.3f}<extra></extra>'
    ))
    
    # GRU trajectory
    fig.add_trace(go.Scatter3d(
        x=gru_3d[:, 0],
        y=gru_3d[:, 1],
        z=gru_3d[:, 2],
        mode='lines+markers',
        line=dict(color=COLORS['gru'], width=4),
        marker=dict(
            size=6,
            color=np.arange(steps),
            colorscale='Purples',
            showscale=False,
            line=dict(color='white', width=1)
        ),
        name='GRU Path',
        hovertemplate='<b>GRU Step %{marker.color}</b><br>Dim1: %{x:.3f}<br>Dim2: %{y:.3f}<br>Dim3: %{z:.3f}<extra></extra>'
    ))
    
    # Add start and end markers
    fig.add_trace(go.Scatter3d(
        x=[esn_3d[0, 0], gru_3d[0, 0]],
        y=[esn_3d[0, 1], gru_3d[0, 1]],
        z=[esn_3d[0, 2], gru_3d[0, 2]],
        mode='markers+text',
        marker=dict(size=12, color=COLORS['success'], symbol='diamond'),
        text=['START', 'START'],
        textposition='top center',
        textfont=dict(size=10, color=COLORS['text']),
        name='Starting Point',
        hovertemplate='<b>Starting Point</b><extra></extra>'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[esn_3d[-1, 0], gru_3d[-1, 0]],
        y=[esn_3d[-1, 1], gru_3d[-1, 1]],
        z=[esn_3d[-1, 2], gru_3d[-1, 2]],
        mode='markers+text',
        marker=dict(size=12, color=COLORS['danger'], symbol='diamond'),
        text=['ESN END', 'GRU END'],
        textposition='bottom center',
        textfont=dict(size=10, color=COLORS['text']),
        name='Final State',
        hovertemplate='<b>Final State</b><extra></extra>'
    ))
    
    # Calculate explained variance
    explained_var = esn_explained  # Use ESN for primary axis labels
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b style='font-size:20px'>Cultural Understanding: 3D Processing Trajectory</b><br>"
                 f"<span style='font-size:14px;color:{COLORS['muted']}'>How ESN and GRU navigate semantic space over {steps} steps</span><br>"
                 f"<span style='font-size:12px;color:{COLORS['muted']}'>ESN variance: {esn_explained[0]:.1%}, {esn_explained[1]:.1%}, {esn_explained[2]:.1%} | "
                 f"GRU variance: {gru_explained[0]:.1%}, {gru_explained[1]:.1%}, {gru_explained[2]:.1%}</span>",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='PC1', backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid']),
            yaxis=dict(title='PC2', backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid']),
            zaxis=dict(title='PC3', backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid']),
            bgcolor=COLORS['card']
        ),
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], family="Inter, system-ui, sans-serif"),
        height=800,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(20, 27, 61, 0.8)',
            bordercolor=COLORS['grid'],
            borderwidth=1
        )
    )
    
    fig.add_annotation(
        text=f"Caption: '{caption[:80]}...'",
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=12, color=COLORS['muted']),
        xanchor='center'
    )
    
    fig.write_html(output_path)
    return output_path


# ============================================================
# 3️⃣ 3D CULTURAL INJECTION: INPUT → RESERVOIR → OUTPUT
# ============================================================

def visualize_3d_cultural_injection(
    caption: str,
    esn: EnhancedESNCulturalController,
    gru: EnhancedGRUCulturalController,
    output_path: str = "viz_server/static/cultural_injection_3d.html"
):
    """
    3D visualization showing the data flow:
    Input Features → Reservoir Processing → Cultural Classification
    """
    # Extract features
    esn_features = esn.extract_features(caption)
    gru_features = gru.extract_features(caption).squeeze().detach().cpu().numpy()
    
    # Process through ESN
    esn.state.fill(0.0)
    esn_mode, esn_conf = esn.predict_mode(caption)
    esn_final_state = esn.state.copy()
    
    # Process through GRU
    gru_mode, gru_conf = gru.predict_mode(caption)
    
    # Create subplots for side-by-side comparison
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('ESN: Echo State Network', 'GRU: Gated Recurrent Unit'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.1
    )
    
    # === ESN VISUALIZATION ===
    # Simple 3D positioning without PCA (since we only have single samples)
    # Layer 0: Input - position at origin
    input_pos_esn = np.array([0, 0, 0])
    
    # Layer 1: Reservoir - position in middle
    # Use mean activation as a simple 3D representation
    reservoir_pos_esn = np.array([
        np.mean(esn_final_state[:len(esn_final_state)//3]),
        np.mean(esn_final_state[len(esn_final_state)//3:2*len(esn_final_state)//3]),
        np.mean(esn_final_state[2*len(esn_final_state)//3:])
    ])
    
    # Layer 2: Output - positions in a plane
    output_positions = np.array([
        [2, -1, 0],      # food_traditional
        [2, 1, 0],       # festival_context
        [3, -1, 0],      # daily_life
        [3, 1, 0]        # generic
    ])
    
    mode_idx = esn.modes.index(esn_mode)
    
    # Add input layer
    fig.add_trace(go.Scatter3d(
        x=[input_pos_esn[0]],
        y=[input_pos_esn[1]],
        z=[input_pos_esn[2]],
        mode='markers+text',
        marker=dict(size=15, color=COLORS['cultural'], symbol='diamond'),
        text=['Input'],
        textposition='top center',
        name='ESN Input',
        hovertemplate='<b>Input Features</b><br>28 cultural dimensions<extra></extra>'
    ), row=1, col=1)
    
    # Add reservoir layer
    fig.add_trace(go.Scatter3d(
        x=[reservoir_pos_esn[0]],
        y=[reservoir_pos_esn[1]],
        z=[reservoir_pos_esn[2]],
        mode='markers+text',
        marker=dict(size=20, color=COLORS['esn'], symbol='circle'),
        text=['Reservoir'],
        textposition='middle center',
        name='ESN Reservoir',
        hovertemplate=f'<b>Reservoir State</b><br>200 neurons active<br>Mean activation: {np.mean(np.abs(esn_final_state)):.3f}<extra></extra>'
    ), row=1, col=1)
    
    # Add output layer
    fig.add_trace(go.Scatter3d(
        x=output_positions[:, 0],
        y=output_positions[:, 1],
        z=output_positions[:, 2],
        mode='markers+text',
        marker=dict(
            size=[20 if i == mode_idx else 10 for i in range(4)],
            color=[COLORS['success'] if i == mode_idx else COLORS['muted'] for i in range(4)],
            symbol='square'
        ),
        text=esn.modes,
        textposition='top center',
        name='ESN Output',
        hovertemplate='<b>%{text}</b><br>Confidence: ' + 
                     '<br>'.join([f"{m}: {esn_conf if m == esn_mode else 0.1:.2f}" for m in esn.modes]) +
                     '<extra></extra>'
    ), row=1, col=1)
    
    # Add flow lines (input → reservoir → output)
    fig.add_trace(go.Scatter3d(
        x=[input_pos_esn[0], reservoir_pos_esn[0]],
        y=[input_pos_esn[1], reservoir_pos_esn[1]],
        z=[input_pos_esn[2], reservoir_pos_esn[2]],
        mode='lines',
        line=dict(color=COLORS['cultural'], width=4, dash='dot'),
        name='ESN Flow',
        hovertemplate='<b>Input → Reservoir</b><extra></extra>',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter3d(
        x=[reservoir_pos_esn[0], output_positions[mode_idx, 0]],
        y=[reservoir_pos_esn[1], output_positions[mode_idx, 1]],
        z=[reservoir_pos_esn[2], output_positions[mode_idx, 2]],
        mode='lines',
        line=dict(color=COLORS['esn'], width=4, dash='dot'),
        name='ESN Output Flow',
        hovertemplate='<b>Reservoir → Output</b><extra></extra>',
        showlegend=False
    ), row=1, col=1)
    
    # === GRU VISUALIZATION ===
    # GRU positions
    input_pos_gru = np.array([0, 0, 0])
    
    # Use GRU hidden state mean as position
    gru_hidden = gru.gru(gru.extract_features(caption), None)[1][-1].squeeze().detach().cpu().numpy()
    hidden_pos_gru = np.array([
        np.mean(gru_hidden[:len(gru_hidden)//3]),
        np.mean(gru_hidden[len(gru_hidden)//3:2*len(gru_hidden)//3]),
        np.mean(gru_hidden[2*len(gru_hidden)//3:])
    ])
    
    gru_mode_idx = gru.modes.index(gru_mode)
    
    # Add GRU input
    fig.add_trace(go.Scatter3d(
        x=[input_pos_gru[0]],
        y=[input_pos_gru[1]],
        z=[input_pos_gru[2]],
        mode='markers+text',
        marker=dict(size=15, color=COLORS['cultural'], symbol='diamond'),
        text=['Input'],
        textposition='top center',
        name='GRU Input',
        hovertemplate='<b>Input Features</b><br>28 cultural dimensions<extra></extra>'
    ), row=1, col=2)
    
    # Add GRU hidden state
    fig.add_trace(go.Scatter3d(
        x=[hidden_pos_gru[0]],
        y=[hidden_pos_gru[1]],
        z=[hidden_pos_gru[2]],
        mode='markers+text',
        marker=dict(size=20, color=COLORS['gru'], symbol='circle'),
        text=['Hidden'],
        textposition='middle center',
        name='GRU Hidden',
        hovertemplate=f'<b>Hidden State</b><br>{gru.hidden_size} units<br>Mean activation: {np.mean(np.abs(gru_hidden)):.3f}<extra></extra>'
    ), row=1, col=2)
    
    # Add GRU output
    fig.add_trace(go.Scatter3d(
        x=output_positions[:, 0],
        y=output_positions[:, 1],
        z=output_positions[:, 2],
        mode='markers+text',
        marker=dict(
            size=[20 if i == gru_mode_idx else 10 for i in range(4)],
            color=[COLORS['success'] if i == gru_mode_idx else COLORS['muted'] for i in range(4)],
            symbol='square'
        ),
        text=gru.modes,
        textposition='top center',
        name='GRU Output',
        hovertemplate='<b>%{text}</b><br>Confidence: ' + 
                     '<br>'.join([f"{m}: {gru_conf if m == gru_mode else 0.1:.2f}" for m in gru.modes]) +
                     '<extra></extra>'
    ), row=1, col=2)
    
    # Add GRU flow lines
    fig.add_trace(go.Scatter3d(
        x=[input_pos_gru[0], hidden_pos_gru[0]],
        y=[input_pos_gru[1], hidden_pos_gru[1]],
        z=[input_pos_gru[2], hidden_pos_gru[2]],
        mode='lines',
        line=dict(color=COLORS['cultural'], width=4, dash='dot'),
        name='GRU Flow',
        hovertemplate='<b>Input → Hidden</b><extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter3d(
        x=[hidden_pos_gru[0], output_positions[gru_mode_idx, 0]],
        y=[hidden_pos_gru[1], output_positions[gru_mode_idx, 1]],
        z=[hidden_pos_gru[2], output_positions[gru_mode_idx, 2]],
        mode='lines',
        line=dict(color=COLORS['gru'], width=4, dash='dot'),
        name='GRU Output Flow',
        hovertemplate='<b>Hidden → Output</b><extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b style='font-size:20px'>Cultural Context Injection: 3D Data Flow</b><br>"
                 f"<span style='font-size:14px;color:{COLORS['muted']}'>Input → Processing → Classification</span>",
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text'], family="Inter, system-ui, sans-serif"),
        height=700,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(20, 27, 61, 0.8)',
            bordercolor=COLORS['grid'],
            borderwidth=1
        )
    )
    
    # Update scene settings
    for row, col in [(1, 1), (1, 2)]:
        fig.update_scenes(
            dict(
                xaxis=dict(backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid'], showticklabels=False),
                yaxis=dict(backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid'], showticklabels=False),
                zaxis=dict(backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid'], showticklabels=False),
                bgcolor=COLORS['card']
            ),
            row=row, col=col
        )
    
    fig.add_annotation(
        text=f"Caption: '{caption[:70]}...'<br>ESN: {esn_mode} ({esn_conf:.2f}) | GRU: {gru_mode} ({gru_conf:.2f})",
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=12, color=COLORS['muted']),
        xanchor='center'
    )
    
    fig.write_html(output_path)
    return output_path


# ============================================================
# 4️⃣ ANIMATED 3D: RESERVOIR EVOLUTION OVER TIME
# ============================================================

def visualize_animated_reservoir_evolution(
    caption: str,
    esn: EnhancedESNCulturalController,
    steps: int = 20,
    sample_neurons: int = 50,
    output_path: str = "viz_server/static/reservoir_evolution_animated.html"
):
    """
    Animated 3D visualization showing how the reservoir state evolves over time.
    Each frame is a processing step.
    """
    # Reset and track all states
    esn.state.fill(0.0)
    x = esn.extract_features(caption)
    
    # Sample neurons
    total_neurons = esn.reservoir_size
    neuron_indices = np.random.choice(total_neurons, min(sample_neurons, total_neurons), replace=False)
    
    # Collect states over time
    state_history = []
    for step in range(steps):
        update = esn.Win @ x + esn.W @ esn.state
        esn.state = (1 - esn.leak_rate) * esn.state + esn.leak_rate * np.tanh(update)
        state_history.append(esn.state[neuron_indices].copy())
    
    state_history = np.array(state_history)  # shape: (steps, sample_neurons)
    
    # Use PCA to reduce to 3D for visualization
    # We need to be careful here - we're reducing neuron dimension, not time
    # Reshape: (steps * sample_neurons) as samples, 1 feature each
    # Actually, we want each neuron across time as a feature
    pca = PCA(n_components=3)
    
    # For 3D visualization: treat each timestep as a point in neuron-space
    # Each point has 'sample_neurons' dimensions, reduce to 3D
    states_3d = pca.fit_transform(state_history)  # shape: (steps, 3)
    
    # Create frames for animation
    frames = []
    
    # We need to show the evolution of neuron activations in 3D space
    # For simplicity, let's use a different approach:
    # Show neurons in a fixed 3D layout, but color changes with activation over time
    
    # Create a fixed 3D layout for neurons (using their indices)
    n_neurons = len(neuron_indices)
    # Arrange neurons in a 3D grid
    grid_size = int(np.ceil(n_neurons ** (1/3)))
    neuron_positions = []
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if idx < n_neurons:
                    neuron_positions.append([i, j, k])
                    idx += 1
    neuron_positions = np.array(neuron_positions[:n_neurons])
    
    for step in range(steps):
        frame_data = go.Scatter3d(
            x=neuron_positions[:, 0],
            y=neuron_positions[:, 1],
            z=neuron_positions[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=state_history[step],
                colorscale='Viridis',
                cmin=state_history.min(),
                cmax=state_history.max(),
                colorbar=dict(
                    title="Activation",
                    thickness=15,
                    len=0.7
                ),
                line=dict(color='white', width=0.5)
            ),
            text=[f"Neuron {neuron_indices[i]}<br>Activation: {state_history[step, i]:.3f}" 
                  for i in range(n_neurons)],
            hovertemplate='<b>%{text}</b><extra></extra>',
            name=f'Step {step}'
        )
        
        frames.append(go.Frame(
            data=[frame_data],
            name=str(step),
            layout=go.Layout(
                title_text=f"Reservoir Evolution - Step {step}/{steps}"
            )
        ))
    
    # Create figure with first frame
    fig = go.Figure(
        data=[frames[0].data[0]],
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=f"<b style='font-size:20px'>ESN Reservoir Evolution (Animated)</b><br>"
                     f"<span style='font-size:14px;color:{COLORS['muted']}'>Cultural processing over {steps} timesteps</span>",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(title='Grid X', backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid']),
                yaxis=dict(title='Grid Y', backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid']),
                zaxis=dict(title='Grid Z', backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid']),
                bgcolor=COLORS['card']
            ),
            paper_bgcolor=COLORS['bg'],
            font=dict(color=COLORS['text'], family="Inter, system-ui, sans-serif"),
            height=800,
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='▶ Play',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
                                mode='immediate'
                            )]
                        ),
                        dict(
                            label='⏸ Pause',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate'
                            )]
                        )
                    ],
                    x=0.1,
                    y=1.15
                )
            ],
            sliders=[dict(
                steps=[dict(
                    args=[[f.name], dict(
                        frame=dict(duration=0, redraw=True),
                        mode='immediate'
                    )],
                    label=f"t={k}",
                    method='animate'
                ) for k, f in enumerate(frames)],
                active=0,
                y=-0.1,
                len=0.9,
                x=0.1
            )]
        )
    )
    
    fig.add_annotation(
        text=f"Caption: '{caption[:70]}...'",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=12, color=COLORS['muted']),
        xanchor='center'
    )
    
    fig.write_html(output_path)
    return output_path