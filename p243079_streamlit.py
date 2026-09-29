import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# Graph, Use Case: Emergency Supply Robot

locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {
        "Main_Corridor": 2.2,
        "Patient_Wing": 4.1
    },

    "Main_Corridor": {
        "Nursing_Station": 2.2
    },

    "Patient_Wing": {
        "Laboratory": 5.0
    },

    "Nursing_Station": {
        "Laboratory": 3.2,
        "Emergency_Ward": 6.0
    },

    "Laboratory": {
        "Emergency_Ward": 3.2
    },

    "Emergency_Ward": {}
}


# Heuristic: straight-line (Euclidean) distance to the goal
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Path reconstruction
def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


# Cost of a path = sum of its edge weights
def path_cost(path):
    return sum(hospital_graph[a][b] for a, b in zip(path, path[1:]))


# GBFS: priority = h(n) only
# Returns (path, total_cost), or (None, None) if the goal is unreachable
def gbfs(start, goal):
    frontier = []
    counter = 0  # tie-breaker so equal priorities pop in insertion order
    heapq.heappush(frontier, (heuristic(start, goal), counter, start))

    came_from = {start: None}
    explored = set()

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, path_cost(path)

        if current in explored:
            continue
        explored.add(current)

        for neighbor in hospital_graph[current]:
            if neighbor not in explored and neighbor not in came_from:
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(
                    frontier, (heuristic(neighbor, goal), counter, neighbor)
                )

    return None, None


# A*: priority = f(n) = g(n) + h(n)
# Returns (path, total_cost), or (None, None) if the goal is unreachable
def a_star(start, goal):
    g_cost = {start: 0}
    came_from = {start: None}
    frontier = []
    counter = 0
    heapq.heappush(frontier, (heuristic(start, goal), counter, start))
    closed = set()

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, current), g_cost[current]

        if current in closed:
            continue
        closed.add(current)

        for neighbor, cost in hospital_graph[current].items():
            tentative_g = g_cost[current] + cost

            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                came_from[neighbor] = current
                counter += 1
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f, counter, neighbor))

    return None, None

##########################################

# Streamlit GUI Code

# Set Page Config
st.set_page_config(
    page_title="Emergency Supply Robot - Informed Search",
    page_icon="🤖",
    layout="wide"
)

# write meaningful title and description for the app
st.title("🤖 Emergency Supply Robot: Informed Search Visualizer")
st.write(
    "A hospital robot must deliver emergency supplies through a network of "
    "corridors. Pick a start node, a goal node and a search algorithm "
    "(**Greedy Best-First Search** or **A\\***) to see the route the robot "
    "chooses, highlighted on the hospital graph, together with its total cost."
)

# define the nodes and their coordinates
nodes = list(hospital_graph.keys())

# create a selectbox for the user to choose the start and goal nodes
start = st.selectbox(
    "Select Initial Node",
    nodes,
    index=nodes.index("Pharmacy")
)

goal = st.selectbox(
    "Select Goal Node",
    nodes,
    index=nodes.index("Emergency_Ward")
)

# create a selectbox for the user to choose the search algorithm
algorithm = st.selectbox(
    "Select Search Algorithm",
    ["GBFS", "A*"]
)

if st.button("Run Search"):

    if algorithm == "GBFS":

        # run the GBFS algorithm with the selected start and goal nodes
        path, cost = gbfs(start, goal)
    else:

        # run the A* algorithm with the selected start and goal nodes
        path, cost = a_star(start, goal)

    if path is None:

        # display a error message indicating that no path was found
        st.error(f"No path found from {start} to {goal}.")

    else:

        # Visualize NetworkX graph

        G = nx.DiGraph()

        for node, neighbors in hospital_graph.items():

            for neighbor, weight in neighbors.items():

                G.add_edge(node, neighbor, weight=weight)
        pos = locations

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        path_edges = list(zip(path, path[1:]))

        # base graph
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color="lightblue", node_size=2600
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color="gray", width=1.5,
            arrows=True, arrowsize=20, node_size=2600
        )

        # highlighted solution path
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=path, node_color="gold", node_size=2600
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=path_edges, edge_color="red", width=4,
            arrows=True, arrowsize=25, node_size=2600
        )

        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax,
            edge_labels=nx.get_edge_attributes(G, "weight"),
            font_size=9
        )

        ax.set_title(
            f"{algorithm} Solution Path"
        )

        ax.axis("off")

        st.pyplot(fig)

        # Display result
        st.subheader("Search Result")

        st.write(
            f"**Algorithm:** {algorithm}"
        )

        st.write(
            f"**Solution Path:** {' → '.join(path)}"
        )

        st.write(
            f"**Total Path Cost:** {cost:.2f}"
        )
