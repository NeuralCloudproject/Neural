import matplotlib.pyplot as plt
import networkx as nx
import sqlite3

class NetworkVisualizer:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def fetch_neuron_data(self, neuron_id: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, threshold, reward FROM neurons WHERE id = ?
        """, (neuron_id,))
        neuron = cursor.fetchone()

        cursor.execute("""
            SELECT target_id, weight FROM synapses WHERE source_id = ?
        """, (neuron_id,))
        connections = cursor.fetchall()

        conn.close()
        return neuron, connections

    def visualize(self, neuron_id: int):
        neuron, connections = self.fetch_neuron_data(neuron_id)

        G = nx.DiGraph()
        G.add_node(f"Neuron-{neuron[0]}", threshold=neuron[1], reward=neuron[2])
        for target_id, weight in connections:
            G.add_edge(f"Neuron-{neuron[0]}", f"Neuron-{target_id}", weight=weight)

        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw_networkx(G, pos, with_labels=True, node_size=700, node_color="lightblue")
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
        plt.title(f"Neuron-{neuron[0]} Network Visualization")
        plt.show()
