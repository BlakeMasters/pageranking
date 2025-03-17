import time
import csv
import re
import networkx as nx
import numpy as np

class GraphParser:
    def __init__(self, file_path, dataset_type):
        self.file_path = file_path
        self.dataset_type = dataset_type

    def clean_numeric_value(self, value):
        return int(re.sub(r'\D', '', value)) if re.search(r'\d', value) else 0

    def parse_csv(self):
        edges = []
        with open(self.file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                row = [item.strip() for item in row if item.strip()]

                if self.dataset_type == "NCAA-FOOTBALL":
                    self.process_ncaa_football(row, edges)
                elif self.dataset_type == "LES-MISERABLES":
                    self.process_les_miserables(row, edges)
                else:
                    self.process_generic_graph(row, edges)

        return edges

    def process_ncaa_football(self, row, edges):
        if len(row) < 4:
            return

        team1, score1, team2, score2 = row[:4]
        overtime = row[4] if len(row) > 4 else None

        score1 = self.clean_numeric_value(score1)
        score2 = self.clean_numeric_value(score2)
        if score1 > score2:
            edges.append((team2, score2, team1, score1, overtime))
        else:
            edges.append((team1, score1, team2, score2, overtime))

    def process_les_miserables(self, row, edges):
        if len(row) != 4:
            return
        node1, label, node2, zero_value = row
        label = self.clean_numeric_value(label)
        zero_value = self.clean_numeric_value(zero_value)
        edges.append((node1, label, node2, zero_value))

    def process_generic_graph(self, row, edges):
        if len(row) < 4:
            return
        node1, value1, node2, value2 = row[:4]
        value1 = self.clean_numeric_value(value1)
        value2 = self.clean_numeric_value(value2)
        edges.append((node1, value1, node2, value2))

class PageRank:
    def __init__(self, graph, epsilon=1e-8, d=0.85, max_iter=100):
        """
        Initializes PageRank.
        :param graph: A networkx.DiGraph.
        :param epsilon: Convergence threshold.
        :param d: Damping factor.
        :param max_iter: Maximum iterations.
        """
        self.graph = graph
        self.epsilon = epsilon
        self.d = d
        self.max_iter = max_iter
        self.rank = {}

    def compute_pagerank(self):
        """
        A[i, j] = 1 / out_degree(j)   if there is an edge from node j to node i
        returns: (iterations, rank_dict)
        """
        nodes = list(self.graph.nodes())
        N = len(nodes)
        if N == 0:
            return 0, {}
        index = {node: i for i, node in enumerate(nodes)}
        A = np.zeros((N, N))
        dangling = np.zeros(N, dtype=bool)
        for node in nodes:
            j = index[node]
            out_deg = self.graph.out_degree(node)
            if out_deg == 0:
                dangling[j] = True
            else:
                for succ in self.graph.successors(node):
                    i = index[succ]
                    A[i, j] = 1.0 / out_deg

        pr = np.ones(N) / N
        for iteration in range(self.max_iter):
            old_pr = pr.copy()
            dangling_sum = old_pr[dangling].sum()
            pr = (1 - self.d) / N + self.d * (A.dot(old_pr) + dangling_sum / N)
            if np.linalg.norm(pr - old_pr, ord=1) < self.epsilon:
                break

        pr_dict = {node: pr[index[node]] for node in nodes}
        self.rank = pr_dict
        return iteration + 1, pr_dict

if __name__ == "__main__":

    filename = "data/NCAA_football.csv"
    dataset_type = "NCAA-FOOTBALL"
    start_read = time.time()
    parser = GraphParser(filename, dataset_type)
    edges = parser.parse_csv()
    end_read = time.time()
    read_time = end_read - start_read
    if dataset_type == "NCAA-FOOTBALL":
        G = nx.DiGraph()
        for edge in edges:
            winner, _, loser, _, _ = edge
            G.add_edge(winner, loser)
    elif dataset_type == "DOLPHINS":
        G = nx.Graph()
        for edge in edges:
        #(dolphin1, val1, dolphin2, val2)
            dolphin1, _, dolphin2, _ = edge
            G.add_edge(dolphin1, dolphin2)
        G_directed = nx.DiGraph()
        for e in G.edges():
            G_directed.add_edge(e[0], e[1])
            G_directed.add_edge(e[1], e[0])
        G = G_directed
    elif dataset_type == "LES-MISERABLES":
        G = nx.Graph()
        for edge in edges:
            char1, label, char2, zero_value = edge
            G.add_edge(char1, char2)
        G_dir = nx.DiGraph()
        for u, v in G.edges():
            G_dir.add_edge(u, v)
            G_dir.add_edge(v, u)
        G = G_dir

    start_process = time.time()
    nx_pr = nx.pagerank(G, alpha=0.85)
    end_process = time.time()
    nx_processing_time = end_process - start_process
    sorted_nx_pr = sorted(nx_pr.items(), key=lambda x: x[1], reverse=True)
    unsorted_nx_pr = sorted(nx_pr.items(), key=lambda x: x[1], reverse=False)

    pagerank_instance = PageRank(G, epsilon=1e-10, d=0.85, max_iter=100)
    start_process_custom = time.time()
    iterations_custom, pr_custom = pagerank_instance.compute_pagerank()
    end_process_custom = time.time()
    custom_processing_time = end_process_custom - start_process_custom
    sorted_custom_pr = sorted(pr_custom.items(), key=lambda x: x[1], reverse=True)

    print("=== NetworkX PageRank (top 10) ===")
    for i, (node, score) in enumerate(sorted_nx_pr[:10], 1):
        print(f"{i}. {node} -> {score}")

    print("\n=== Custom PageRank (top 10) ===")
    for i, (node, score) in enumerate(sorted_custom_pr[:10], 1):
        print(f"{i}. {node} -> {score}")

    print(f"\nNumber of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Read time: {read_time:.6f} seconds")
    print(f"NetworkX PageRank processing time: {nx_processing_time:.6f} seconds")
    print(f"Custom PageRank processing time: {custom_processing_time:.6f} seconds")
    print(f"Custom PageRank iterations: {iterations_custom}")
