import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import HeteroEmbedding


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_0 for transform the node's own feature
        self.weight0 = nn.Linear(in_size, out_size)
        
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, g, feat_dict, eweight_dict=None):
        # The input is a dictionary of node features for each type
        funcs = {}
        if eweight_dict is not None:
            # Store the sigmoid of edge weights
            g.edata['_edge_weight'] = eweight_dict
                
        for srctype, etype, dsttype in g.canonical_etypes:
            # Compute h_0 = W_0 * h
            h0 = self.weight0(feat_dict[dsttype])
            g.nodes[dsttype].data['h0'] = h0
            
            # Compute h_r = W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            g.nodes[srctype].data['Wh_%s' % etype] = Wh
            
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            if eweight_dict is not None:
                msg_fn = fn.u_mul_e('Wh_%s' % etype, '_edge_weight', 'm')
            else:
                msg_fn = fn.copy_u('Wh_%s' % etype, 'm')
                
            funcs[(srctype, etype, dsttype)] = (msg_fn, fn.mean('m', 'h'))

        def apply_func(nodes):
            h = nodes.data['h'] + nodes.data['h0']
            return {'h': h}
            
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        g.multi_update_all(funcs, 'sum', apply_func)
        # g.multi_update_all(funcs, 'sum')

        # return the updated node feature dictionary
        return {ntype : g.nodes[ntype].data['h'] for ntype in g.ntypes}
    
class HeteroRGCN(nn.Module):
    def __init__(self, g, emb_dim, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()        
        self.emb = HeteroEmbedding({ntype : g.num_nodes(ntype) for ntype in g.ntypes}, emb_dim)
       
        self.layer1 = HeteroRGCNLayer(emb_dim, hidden_size, g.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, g.etypes)

    def forward(self, g_query, feat_nids=None, eweight_dict=None):
        if feat_nids is None:
            feat_dict = self.emb.weight
        else:
            feat_dict = self.emb(feat_nids)

        
        h_dict = self.layer1(g_query, feat_dict, eweight_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
       
        
        h_dict = self.layer2(g_query, h_dict, eweight_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        
        # h_dict = {k : F.dropout(h,self.drop_gcn_1 ) for k, h in h_dict.items()}
        
        return h_dict