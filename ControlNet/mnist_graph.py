import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data


class MNISTGraph:
    def __init__(self):

        self.digit_embedding = {
            0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }

        self.edge_embedding = {
            "left of": [1, 0, 0, 0],
            "right of": [0, 1, 0, 0],
            "above": [0, 0, 1, 0],
            "below": [0, 0, 0, 1]
        }
    
    def create_graph(self, prompt):
        
        # Nodes in your graph (let's assume that the node features are one-hot encoded digits)
        # Nodes: 7, 6, 3
        x = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 7
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 6
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], # 3
                        dtype=torch.float)

        # Edges: 7 (node 0) is below 6 (node 1), 6 (node 1) is left of 3 (node 2)
        edge_index = torch.tensor([[0, 1,    # 7 below 6
                                    1, 2],   # 6 left of 3
                                [1, 2,    # 6 above 7
                                    2, 1]],  # 3 right of 6
                                dtype=torch.long)

        # Edge types: 3 ("below"), 0 ("left of"), 2 ("above"), 1 ("right of")
        edge_type = torch.tensor([3, 0, 2, 1], dtype=torch.long)

        # Create PyTorch Geometric graph data
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        print(data)


    def visualise(self):
        self.G.add_edge('7', '6', relationship='below')
        self.G.add_edge('6', '3', relationship='left of')
        nx.draw(self.G, with_labels=True)
        plt.show()      
  

if __name__ == '__main__':
    m = MNISTGraph()
    m.visualise()