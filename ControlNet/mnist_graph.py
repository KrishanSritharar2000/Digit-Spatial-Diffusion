import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric import conv
import re

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
            "left of": 0,
            "right of": 1,
            "above": 2,
            "below": 3
        }

        self.opp_edge_embedding = {
            "left of": 1,
            "right of": 0,
            "above": 3,
            "below": 2
        }

        self.digit_regex = r'\b\d\b'
        self.relationship_regex = r'\b(left of|right of|above|below)\b'
    
    def get_digit_and_relationship(self, prompt):
        prompt = prompt.strip()
        p_digits = list(map(int, re.findall(self.digit_regex, prompt)))
        p_relationships = re.findall(self.relationship_regex, prompt)
        p_relationships_and_digits = [(p_digits[i], p_relationships[i], p_digits[i + 1]) for i in range(len(p_relationships))]
        return p_digits, p_relationships, p_relationships_and_digits

    def create_graph(self, prompt):
        digits, relationships, relationships_and_digits = self.get_digit_and_relationship(prompt)

        digit_embeddings = []
        edge_index_embeddings = [[],[]]
        edge_type_embeddings = []

        first_loop = True
        node_counter = 0

        for triple in relationships_and_digits:
            node1, node2 = triple[0], triple[2]
            relationship = triple[1]

            if first_loop:
                digit_embeddings.append(self.digit_embedding[node1])
                first_loop = False
            digit_embeddings.append(self.digit_embedding[node2])
            
            edge_index_embeddings[0].extend([node_counter, node_counter + 1])
            edge_index_embeddings[1].extend([node_counter + 1 , node_counter])

            edge_type_embeddings.append(self.edge_embedding[relationship])
            edge_type_embeddings.append(self.opp_edge_embedding[relationship])
            
            node_counter += 1

        x = torch.tensor(digit_embeddings, dtype=torch.float)
        edge_index = torch.tensor(edge_index_embeddings, dtype=torch.long)
        edge_type = torch.tensor(edge_type_embeddings, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        return data
    


        # # Nodes in your graph (let's assume that the node features are one-hot encoded digits)
        # # Nodes: 7, 6, 3
        # x = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 7
        #                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 6
        #                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], # 3
        #                 dtype=torch.float)

        # # Edges: 7 (node 0) is below 6 (node 1), 6 (node 1) is left of 3 (node 2)
        # edge_index = torch.tensor([[0, 1,    # 7 below 6
        #                             1, 2],   # 6 left of 3
        #                         [1, 2,    # 6 above 7
        #                             2, 1]],  # 3 right of 6
        #                         dtype=torch.long)

        # # Edge types: 3 ("below"), 0 ("left of"), 2 ("above"), 1 ("right of")
        # edge_type = torch.tensor([3, 0, 2, 1], dtype=torch.long)

        # # Create PyTorch Geometric graph data
        # data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        # print(data)


if __name__ == '__main__':
    g = MNISTGraph()
    data = g.create_graph("7 below 6 left of 3")
    # m.visualise()

    import torch
    from torch.nn import Linear
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data

    class GNN(torch.nn.Module):
        def __init__(self):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(10, 64)  # Assuming each node has a 10-dimensional feature vector
            self.conv2 = GCNConv(64, 128)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)

            return x

    model = GNN()
    out = model(data)  # out is the embeddings of nodes
    RGCN = RGCNConv(10, 64, 4, num_relations=4)