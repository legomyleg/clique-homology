import networkx as nx
import plotly.graph_objects as go

G = nx.random_geometric_graph(50, 0.25)
pos = nx.spring_layout(G)

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# Create an empty list to hold the weights (degrees)
node_weights = []

# Loop through the networkx graph and grab the degree of each node
for node, degree in G.degree():
    node_weights.append(degree)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=10,
        color=node_weights,
        line_width=2
    )
)

fig = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                title='NetworkX Graph using Plotly',
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
)

fig.show()