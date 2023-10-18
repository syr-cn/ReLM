import torch
import torch.nn as nn

import sklearn
import dgl
import dgllife
from dgllife.model import MPNNGNN
from dgl.nn.pytorch.glob import SumPooling

from .model_utils import pair_atom_feats, unbatch_mask, unbatch_feats, Global_Reactivity_Attention

class LocalRetro(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats,
                 edge_hidden_feats,
                 num_step_message_passing,
                 attention_heads,
                 attention_layers,
                 AtomTemplate_n, 
                 BondTemplate_n,
                 device='cpu'):
        super(LocalRetro, self).__init__()
                
        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        self.linearB = nn.Linear(node_out_feats*2, node_out_feats)

        self.att = Global_Reactivity_Attention(node_out_feats, attention_heads, attention_layers)
        
        self.atom_linear =  nn.Sequential(
                            nn.Linear(node_out_feats, node_out_feats), 
                            nn.ReLU(), 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, AtomTemplate_n+1))
        self.bond_linear =  nn.Sequential(
                            nn.Linear(node_out_feats, node_out_feats), 
                            nn.ReLU(), 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, BondTemplate_n+1))
        self.pooling = SumPooling()
        self.device=device

    def forward_old(self, g, node_feats, edge_feats):
        node_feats = self.mpnn(g, node_feats, edge_feats)
        atom_feats = node_feats
        bond_feats = self.linearB(pair_atom_feats(g, node_feats))
        edit_feats, mask = unbatch_mask(g, atom_feats, bond_feats)
        attention_score, edit_feats = self.att(edit_feats, mask)

        atom_feats, bond_feats = unbatch_feats(g, edit_feats)
        atom_outs = self.atom_linear(atom_feats) 
        bond_outs = self.bond_linear(bond_feats) 

        return atom_outs, bond_outs, attention_score

    @torch.no_grad()
    def forward(self, g):
        g = g.to(self.device)
        node_feats = g.ndata.pop('h').to(g.device)
        edge_feats = g.edata.pop('e').to(g.device)
        node_feats = self.mpnn(g, node_feats, edge_feats)
        g_feat = self.pooling(g, node_feats)
        return g_feat # [B, H]


