import pickle
import random
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from slugify import slugify

random.seed(69420)

def jaccard(s1, s2):
    if not s1 or not s2:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))


class DataLoader:
    """
    Responsible for loading the specific Eurovision pickle files and preparing the graph.
    """

    @staticmethod
    def load_eurovision_data(
        graph_path: str = "G_esc_2018.pkl",
        pos_path: str = "pos_geo.pkl",
        flags_path: str = "flags.pkl",
    ) -> Tuple[nx.DiGraph, Dict, Dict]:
        """
        Loads the graph and metadata from pickle files as specified in the appendix.
        """
        try:
            with open(graph_path, "rb") as f:
                G = pickle.load(f)

            with open(pos_path, "rb") as f:
                pos_geo = pickle.load(f)

            with open(flags_path, "rb") as f:
                flags = pickle.load(f)

            for u, v, data in G.edges(data=True):
                if "points" in data:
                    data["weight"] = data["points"]
                else:
                    data["weight"] = 0

            return G, pos_geo, flags

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find one of the data files: {e}")


class GraphTransformer:
    """
    Responsible for creating undirected/weighted variants (Tasks c, d, e).
    """

    @staticmethod
    def to_undirected_bidirectional(G: nx.DiGraph) -> nx.Graph:
        """Task c: Edge exists only if u->v AND v->u."""
        G_u1 = nx.Graph()
        G_u1.add_nodes_from(G.nodes)
        for u, v in G.edges:
            if G.has_edge(v, u):
                G_u1.add_edge(u, v)
        return G_u1

    @staticmethod
    def to_undirected_union(G: nx.DiGraph) -> nx.Graph:
        """Task d: Edge exists if u->v OR v->u."""
        return nx.Graph(G)

    @staticmethod
    def to_undirected_weighted_sum(G: nx.DiGraph) -> nx.Graph:
        """Task e: Weight is sum of unidirectional points."""
        G_u3 = nx.Graph()
        G_u3.add_nodes_from(G.nodes())
        processed_pairs = set()

        for u, v, data in G.edges(data=True):
            pair = tuple(sorted((u, v)))
            if pair in processed_pairs:
                continue

            w_uv = data.get("weight", 0)
            w_vu = G.get_edge_data(v, u, default={"weight": 0}).get("weight", 0)

            total_weight = w_uv + w_vu
            if total_weight > 0:
                G_u3.add_edge(u, v, weight=total_weight)

            processed_pairs.add(pair)
        return G_u3


class RandomGraphGenerator:
    """
    Responsible for generating random graph variants (Task a).
    """

    @staticmethod
    def generate_random_configuration(G_original: nx.DiGraph) -> nx.DiGraph:
        nodes = list(G_original.nodes())
        weights = [d["weight"] for u, v, d in G_original.edges(data=True)]

        G_rand = nx.DiGraph()
        G_rand.add_nodes_from(nodes)

        possible_edges = list(combinations(nodes, 2))
        possible_edges += [(v, u) for u, v in possible_edges]

        if len(weights) > len(possible_edges):
            pass

        selected_edges = random.sample(possible_edges, len(weights))

        for (u, v), w in zip(selected_edges, weights):
            G_rand.add_edge(u, v, weight=w)

        return G_rand


class NetworkAnalyzer:
    @staticmethod
    def get_weighted_clustering_coefficient(G: nx.Graph) -> float:
        return nx.average_clustering(G, weight="weight")

    @staticmethod
    def get_in_degree_distribution(G: nx.DiGraph) -> List[float]:
        degrees = dict(G.in_degree(weight="weight"))
        return list(degrees.values())

    @staticmethod
    def get_girvan_newman_communities(
        G: nx.Graph, num_communities: int = 2
    ) -> Tuple[Set, Set]:
        comp_gen = nx.community.girvan_newman(G)
        try:
            communities = next(comp_gen)

            # Output two largest communities
            if len(communities) > num_communities:
                communities = sorted(communities, key=len, reverse=True)[
                    :num_communities
                ]

            return tuple(communities)
        except StopIteration:
            return tuple(nx.connected_components(G))

    @staticmethod
    def get_kernighan_lin_communities(G: nx.Graph) -> Tuple[Set, Set]:
        return nx.community.kernighan_lin_bisection(G, weight="weight")

    @staticmethod
    def find_community_cores(partitions: List[Tuple[Set, Set]]) -> Tuple[Set, Set]:
        """
        Task f: Form maximal intersections to find "cores".
        Jaccard alignment to ensure we intersect the correct matching groups.
        """
        if not partitions:
            return set(), set()

        core_A = partitions[0][0]
        core_B = partitions[0][1]

        for next_part in partitions[1:]:
            part_0, part_1 = next_part

            sim_straight = jaccard(core_A, part_0) + jaccard(core_B, part_1)

            sim_crossed = jaccard(core_A, part_1) + jaccard(core_B, part_0)

            if sim_straight >= sim_crossed:
                # Keep orientation
                core_A = core_A.intersection(part_0)
                core_B = core_B.intersection(part_1)
            else:
                # Swap orientation
                core_A = core_A.intersection(part_1)
                core_B = core_B.intersection(part_0)

        return core_A, core_B


class Visualizer:
    @staticmethod
    def compare_degree_distributions(degrees_real, degrees_random, bins=15):
        """
        Plots histograms optimized for LaTeX reports (approx \textwidth).
        """
        plt.rcParams.update({"font.size": 12})

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.hist(degrees_real, bins=bins, color="skyblue", edgecolor="black", alpha=0.9)
        plt.title("Real Graph In-Degree", fontsize=14, fontweight="bold")
        plt.xlabel("Weighted In-Degree", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)

        plt.subplot(1, 2, 2)
        plt.hist(
            degrees_random, bins=bins, color="salmon", edgecolor="black", alpha=0.9
        )
        plt.title("Random Graph In-Degree", fontsize=14, fontweight="bold")
        plt.xlabel("Weighted In-Degree", fontsize=12)

        plt.tight_layout()
        plt.savefig(f"imgs/degree_distribution.pdf", bbox_inches="tight")
        plt.show()

    @staticmethod
    def draw_eurovision_map_with_communities(
        G: nx.Graph,
        pos_geo: Dict,
        flags: Dict,
        communities: Optional[Tuple[Set, Set]] = None,
        title: str = "Eurovision Network",
    ):
        plt.rcParams.update({"font.size": 14})

        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        plt.axis("off")

        plt.title(title, fontsize=18, pad=20)

        trans = ax.transData.transform
        trans2 = plt.gcf().transFigure.inverted().transform

        styles = ["dotted", "dashdot", "dashed", "solid"]
        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 0)
            width = weight / 24.0
            style = styles[min(int(width * 3), 3)]

            if width > 0.3:
                nx.draw_networkx_edges(
                    G,
                    pos_geo,
                    edgelist=[(u, v)],
                    width=width,
                    style=style,
                    edge_color="#D3D3D3",
                    alpha=0.6,
                    arrows=True,
                    arrowsize=15,
                )

        for node in G.nodes():
            if node not in flags:
                continue

            flag_img = mpl.image.imread(flags[node])
            (x, y) = pos_geo[node]
            xx, yy = trans((x, y))
            xa, ya = trans2((xx, yy))

            imsize = 0.045

            country_ax = plt.axes(
                [xa - imsize / 2.0, ya - imsize / 2.0, imsize, imsize]
            )
            country_ax.imshow(flag_img)
            country_ax.set_aspect("equal")
            country_ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            if communities:
                c1, c2 = communities
                if node in c1:
                    border_color = "lime"
                    linewidth = 2.5
                    zorder = 10
                elif node in c2:
                    border_color = "fuchsia"
                    linewidth = 2.5
                    zorder = 10
                else:
                    border_color = "silver"
                    linewidth = 1
                    zorder = 1

                for spine in country_ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(linewidth)
                    spine.set_zorder(zorder)
            else:
                for spine in country_ax.spines.values():
                    spine.set_visible(False)

        plt.savefig(f"imgs/eurovision_map_{slugify(title)}.pdf", bbox_inches="tight")
        plt.show()
