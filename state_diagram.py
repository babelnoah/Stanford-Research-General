import matplotlib.pyplot as plt
import networkx as nx

def draw_detailed_state_diagram():
    G = nx.DiGraph()

    # Define states
    states = ["x1", "x2", "x3", "x4"]

    # Add nodes
    for state in states:
        G.add_node(state)

    # Define some transitions based on lambda coefficients (This is representative; adjust as needed)
    transitions = {
        "x1": {"x2": "λ12", "x3": "λ13", "x4": "λ14"},
        "x2": {"x1": "λ21", "x3": "λ23", "x4": "λ24"},
        "x3": {"x1": "λ31", "x2": "λ32", "x4": "λ34"},
        "x4": {}  # As per your definition, x4 is derived, so no direct transitions
    }

    # Add transitions to graph
    for start, ends in transitions.items():
        for end, label in ends.items():
            G.add_edge(start, end, label=label)

    # Position nodes using a shell layout for better separation
    shell_pos = [["x1", "x2", "x3", "x4"]]
    pos = nx.shell_layout(G, shell_pos)

    # Draw the graph
    plt.figure(figsize=(12, 10))

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=15, font_weight='bold',
            edge_color='gray', width=2, alpha=0.6)

    # Draw edge labels based on lambda coefficients
    edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_weight='bold')

    plt.title("Detailed State Transition Diagram")
    plt.show()

draw_detailed_state_diagram()
