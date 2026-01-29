import networkit as nk
from networkit.graph import Graph

TEST_CSV = "docs/data/C-elegans-frontal.csv"
HEADERS = ["FromNodeId,ToNodeId"]


def csv_to_graph(file: str) -> Graph:
    """
    Takes a file path as a string and returns a networkit `Graph` object. 
    Handles any CSV file that uses a header found in the list of `HEADERS` and has format for nodes: "from_node,to_node".
    Assumes that vertices have IDs 0 to n-1 with n being the number of vertices. If `min(vertex_list)` is not `0`, 
    function returns an error.
    """
    vertex_set: set[int] = set()
    edge_set: set[tuple[int, int]] = set()
    
    with open(file, newline="", encoding="utf-8") as f:
        
        for line in f:
            if line.strip() in HEADERS:
                f.readline()
                break
        else:
            raise ValueError("Header not found.")
        
        for line in f:
            if line.strip() == "" or len(line.split(",")) != 2:
                continue
            line: str = line.strip()
            nodes: list = line.split(",")
            vertex_set.update(nodes)
            edge_set.add(sorted(tuple(nodes)))
            
    if min(vertex_set) != 0 or max(vertex_set) != len(vertex_set) - 1:
        raise ValueError("Nodes are not enumerated 0 to n-1.")
        
    g = Graph(n=max(vertex_set), )
    
    
                
    