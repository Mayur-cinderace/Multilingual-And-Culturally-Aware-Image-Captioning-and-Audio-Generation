# esn_gru_visualization.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

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
    'grid': '#1e2749'
}

def create_base_layout(title, subtitle=None):
    """Create consistent, beautiful base layout"""
    return dict(
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], family="Inter, system-ui, sans-serif", size=13),
        title=dict(
            text=f"<b style='font-size:20px'>{title}</b>" + 
                 (f"<br><span style='font-size:14px;color:{COLORS['muted']}'>{subtitle}</span>" if subtitle else ""),
            x=0.5,
            xanchor='center',
            y=0.97,
            yanchor='top'
        ),
        margin=dict(l=60, r=40, t=100, b=60),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            bgcolor='rgba(20, 27, 61, 0.8)',
            bordercolor=COLORS['grid'],
            borderwidth=1,
            font=dict(size=11)
        )
    )

# ============================================================
# 1️⃣ ACTIVATION HEATMAP — Clear temporal patterns
# ============================================================

def visualize_esn_activation_flow(
    caption: str,
    esn: EnhancedESNCulturalController,
    steps: int = 10,
    top_neurons: int = 50,
    output_path: str = "viz_server/static/activation_flow.html"
):
    """
    Show how specific neurons activate over time.
    Much clearer than 3D scatter: shows WHICH neurons matter WHEN.
    """
    esn.state.fill(0.0)
    x = esn.extract_features(caption)
    
    # Track activation history
    activation_history = []
    
    for step in range(steps):
        update = esn.Win @ x + esn.W @ esn.state
        esn.state = (1 - esn.leak_rate) * esn.state + esn.leak_rate * np.tanh(update)
        activation_history.append(esn.state.copy())
    
    # Convert to matrix: [steps x neurons]
    activation_matrix = np.array(activation_history)
    
    # Find most active neurons across all time
    neuron_importance = np.max(np.abs(activation_matrix), axis=0)
    top_indices = np.argsort(neuron_importance)[-top_neurons:]
    
    # Extract just the important neurons
    heatmap_data = activation_matrix[:, top_indices].T
    
    # Create heatmap
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=[f"t={i}" for i in range(steps)],
        y=[f"N{idx}" for idx in top_indices],
        colorscale=[
            [0, '#1e2749'],
            [0.3, '#3b82f6'],
            [0.6, '#8b5cf6'],
            [1, '#f59e0b']
        ],
        colorbar=dict(
            title="Activation",
            thickness=15,
            len=0.7
        ),
        hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Activation: %{z:.3f}<extra></extra>'
    ))
    
    layout = create_base_layout(
        "Cultural Meaning Formation Timeline",
        "Tracking which neural patterns activate as the model processes cultural context"
    )
    
    layout.update(dict(
        xaxis=dict(
            title="Processing Steps",
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title=f"Top {top_neurons} Cultural Neurons",
            showgrid=False,
            zeroline=False
        ),
        height=600
    ))
    
    fig.update_layout(layout)
    
    # Add annotation explaining the pattern
    fig.add_annotation(
        text=f"Input: '{caption[:60]}...'",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12, color=COLORS['muted']),
        xanchor='center'
    )
    
    fig.write_html(output_path)
    return output_path

# ============================================================
# 2️⃣ FEATURE IMPORTANCE — What drives cultural understanding?
# ============================================================

def visualize_cultural_feature_importance(
    caption: str,
    esn: EnhancedESNCulturalController,
    gru: EnhancedGRUCulturalController,
    output_path: str = "viz_server/static/feature_importance.html"
):
    """
    Compare what each model pays attention to.
    Bar chart is clearer than 3D scatter for interpretation.
    """
    # Get final states
    esn.predict_mode(caption)
    esn_state = esn.state
    
    gru_features = gru.extract_features(caption).squeeze().detach().cpu().numpy()
    
    # Compute feature importance (top activated dimensions)
    esn_importance = np.abs(esn_state)
    gru_importance = np.abs(gru_features)
    
    # Get top features
    n_show = 20
    esn_top_idx = np.argsort(esn_importance)[-n_show:]
    gru_top_idx = np.argsort(gru_importance)[-n_show:]
    
    # Create side-by-side comparison
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('ESN: Distributed Cultural Memory', 'GRU: Compressed Representation'),
        horizontal_spacing=0.12
    )
    
    # ESN bars
    fig.add_trace(
        go.Bar(
            x=esn_importance[esn_top_idx],
            y=[f"Feature {i}" for i in esn_top_idx],
            orientation='h',
            marker=dict(
                color=esn_importance[esn_top_idx],
                colorscale='Blues',
                showscale=False
            ),
            name='ESN',
            hovertemplate='<b>%{y}</b><br>Activation: %{x:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # GRU bars
    fig.add_trace(
        go.Bar(
            x=gru_importance[gru_top_idx],
            y=[f"Feature {i}" for i in gru_top_idx],
            orientation='h',
            marker=dict(
                color=gru_importance[gru_top_idx],
                colorscale='Purples',
                showscale=False
            ),
            name='GRU',
            hovertemplate='<b>%{y}</b><br>Activation: %{x:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    layout = create_base_layout(
        "Cultural Understanding: What Each Model Pays Attention To",
        "Comparing feature activation patterns between ESN and GRU architectures"
    )
    
    layout.update(dict(
        height=700,
        showlegend=False,
        xaxis=dict(title="Activation Strength", gridcolor=COLORS['grid']),
        xaxis2=dict(title="Activation Strength", gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
        yaxis2=dict(gridcolor=COLORS['grid'])
    ))
    
    fig.update_layout(layout)
    
    fig.add_annotation(
        text=f"Input: '{caption[:70]}...'",
        xref="paper", yref="paper",
        x=0.5, y=-0.08,
        showarrow=False,
        font=dict(size=12, color=COLORS['muted']),
        xanchor='center'
    )
    
    fig.write_html(output_path)
    return output_path

# ============================================================
# 3️⃣ CULTURAL DIMENSIONS — Interpretable 2D projection
# ============================================================

def visualize_cultural_dimensions(
    captions: list,
    esn: EnhancedESNCulturalController,
    output_path: str = "viz_server/static/cultural_space.html"
):
    """
    2D projection showing how different captions cluster in cultural space.
    Much more interpretable than 3D.
    """
    # Get embeddings for each caption
    embeddings = []
    for caption in captions:
        esn.predict_mode(caption)
        embeddings.append(esn.state.copy())
    
    embeddings = np.array(embeddings)
    
    # Use t-SNE for better clustering visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(captions)-1))
    coords_2d = tsne.fit_transform(embeddings)
    
    # Calculate "cultural intensity" as magnitude
    intensities = np.linalg.norm(embeddings, axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        mode='markers+text',
        marker=dict(
            size=intensities * 5,
            color=intensities,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Cultural<br>Intensity",
                thickness=15,
                len=0.7
            ),
            line=dict(width=1, color='white')
        ),
        text=[f"Caption {i+1}" for i in range(len(captions))],
        textposition="top center",
        textfont=dict(size=10, color=COLORS['text']),
        hovertemplate='<b>Caption %{text}</b><br>%{hovertext}<extra></extra>',
        hovertext=[f"{c[:100]}..." for c in captions]
    ))
    
    layout = create_base_layout(
        "Cultural Semantic Space",
        "How different cultural contexts cluster in the model's understanding"
    )
    
    layout.update(dict(
        height=700,
        xaxis=dict(
            title="Cultural Dimension 1",
            showgrid=True,
            gridcolor=COLORS['grid'],
            zeroline=True,
            zerolinecolor=COLORS['grid']
        ),
        yaxis=dict(
            title="Cultural Dimension 2",
            showgrid=True,
            gridcolor=COLORS['grid'],
            zeroline=True,
            zerolinecolor=COLORS['grid']
        ),
        showlegend=False
    ))
    
    fig.update_layout(layout)
    
    fig.write_html(output_path)
    return output_path

# ============================================================
# 4️⃣ COUNTERFACTUAL COMPARISON — Clear delta visualization
# ============================================================

def visualize_counterfactual_impact(
    original_caption: str,
    modified_captions: dict,  # {label: modified_caption}
    esn: EnhancedESNCulturalController,
    output_path: str = "viz_server/static/counterfactual.html"
):
    """
    Show how interpretation changes with different cultural framings.
    Radar chart makes differences crystal clear.
    """
    # Get baseline
    esn.predict_mode(original_caption)
    baseline_state = esn.state.copy()
    
    # Define interpretable cultural dimensions
    dim_names = [
        "Ritual Significance",
        "Social Cohesion",
        "Temporal Awareness",
        "Sacred vs Mundane",
        "Collective Identity",
        "Emotional Intensity"
    ]
    
    def extract_dimensions(state):
        """Extract interpretable dimensions from reservoir state"""
        n_dims = len(dim_names)
        chunk_size = len(state) // n_dims
        
        dims = []
        for i in range(n_dims):
            start = i * chunk_size
            end = start + chunk_size
            dims.append(np.mean(np.abs(state[start:end])))
        
        return dims
    
    baseline_dims = extract_dimensions(baseline_state)
    
    # Create radar chart
    fig = go.Figure()
    
    # Add baseline
    fig.add_trace(go.Scatterpolar(
        r=baseline_dims + [baseline_dims[0]],
        theta=dim_names + [dim_names[0]],
        fill='toself',
        name='Original',
        line=dict(color=COLORS['primary'], width=2),
        fillcolor=f"rgba(59, 130, 246, 0.2)"
    ))
    
    # Add counterfactuals
    colors = [COLORS['secondary'], COLORS['accent'], COLORS['success'], COLORS['danger']]
    
    for idx, (label, mod_caption) in enumerate(modified_captions.items()):
        esn.predict_mode(mod_caption)
        mod_state = esn.state
        mod_dims = extract_dimensions(mod_state)
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatterpolar(
            r=mod_dims + [mod_dims[0]],
            theta=dim_names + [dim_names[0]],
            fill='toself',
            name=label,
            line=dict(color=color, width=2),
            fillcolor=f"rgba{tuple(list(bytes.fromhex(color[1:])) + [0.15])}"
        ))
    
    layout = create_base_layout(
        "Counterfactual Cultural Analysis",
        "How cultural interpretation shifts across different contextual frames"
    )
    
    layout.update(dict(
        height=700,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(baseline_dims) * 1.2],
                gridcolor=COLORS['grid']
            ),
            bgcolor=COLORS['card']
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.25
        )
    ))
    
    fig.update_layout(layout)
    
    fig.write_html(output_path)
    return output_path

# ============================================================
# 5️⃣ CONVERGENCE ANALYSIS — Show stability
# ============================================================

def visualize_convergence(
    caption: str,
    esn: EnhancedESNCulturalController,
    steps: int = 50,
    output_path: str = "viz_server/static/convergence.html"
):
    """
    Show how the model's interpretation stabilizes over time.
    Line chart is perfect for this.
    """
    esn.state.fill(0.0)
    x = esn.extract_features(caption)
    
    metrics = {
        'state_norm': [],
        'state_change': [],
        'entropy': []
    }
    
    prev_state = esn.state.copy()
    
    for step in range(steps):
        update = esn.Win @ x + esn.W @ esn.state
        esn.state = (1 - esn.leak_rate) * esn.state + esn.leak_rate * np.tanh(update)
        
        # Calculate metrics
        metrics['state_norm'].append(np.linalg.norm(esn.state))
        metrics['state_change'].append(np.linalg.norm(esn.state - prev_state))
        
        # Pseudo-entropy based on activation distribution
        abs_state = np.abs(esn.state)
        if abs_state.sum() > 0:
            p = abs_state / abs_state.sum()
            entropy = -np.sum(p * np.log(p + 1e-10))
            metrics['entropy'].append(entropy)
        else:
            metrics['entropy'].append(0)
        
        prev_state = esn.state.copy()
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Activation Magnitude',
            'State Change (Convergence)',
            'Activation Entropy (Diversity)'
        ),
        vertical_spacing=0.12
    )
    
    # State norm
    fig.add_trace(
        go.Scatter(
            x=list(range(steps)),
            y=metrics['state_norm'],
            mode='lines',
            line=dict(color=COLORS['primary'], width=2),
            name='Magnitude',
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.2)'
        ),
        row=1, col=1
    )
    
    # State change
    fig.add_trace(
        go.Scatter(
            x=list(range(steps)),
            y=metrics['state_change'],
            mode='lines',
            line=dict(color=COLORS['success'], width=2),
            name='Change',
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.2)'
        ),
        row=2, col=1
    )
    
    # Entropy
    fig.add_trace(
        go.Scatter(
            x=list(range(steps)),
            y=metrics['entropy'],
            mode='lines',
            line=dict(color=COLORS['accent'], width=2),
            name='Entropy',
            fill='tozeroy',
            fillcolor='rgba(245, 158, 11, 0.2)'
        ),
        row=3, col=1
    )
    
    layout = create_base_layout(
        "Cultural Meaning Convergence Analysis",
        "How the model's interpretation stabilizes over recurrent processing steps"
    )
    
    layout.update(dict(
        height=900,
        showlegend=False,
        xaxis3=dict(title="Processing Steps", gridcolor=COLORS['grid']),
    ))
    
    for i in [1, 2, 3]:
        fig.update_yaxes(gridcolor=COLORS['grid'], row=i, col=1)
        fig.update_xaxes(gridcolor=COLORS['grid'], row=i, col=1)
    
    fig.update_layout(layout)
    
    fig.write_html(output_path)
    return output_path