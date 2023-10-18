import os
import numpy as np

import time
import torch
import torch.nn as nn

from utils import predict

import dgl

def get_id_template(a, class_n):
    class_n = class_n # no template
    edit_idx = a//class_n
    template = a%class_n
    return (edit_idx, template)

def output2edit(out, top_num):
    class_n = out.size(-1)
    readout = out.cpu().detach().numpy()
    readout = readout.reshape(-1)
    output_rank = np.flip(np.argsort(readout))
    output_rank = [r for r in output_rank if get_id_template(r, class_n)[1] != 0][:top_num]
    
    selected_edit = [get_id_template(a, class_n) for a in output_rank]
    selected_proba = [readout[a] for a in output_rank]
     
    return selected_edit, selected_proba
    
def combined_edit(graph, atom_out, bond_out, top_num):
    edit_id_a, edit_proba_a = output2edit(atom_out, top_num)
    edit_id_b, edit_proba_b = output2edit(bond_out, top_num)
    edit_id_c = edit_id_a + edit_id_b
    edit_type_c = ['a'] * top_num + ['b'] * top_num
    edit_proba_c = edit_proba_a + edit_proba_b
    edit_rank_c = np.flip(np.argsort(edit_proba_c))[:top_num]
    edit_type_c = [edit_type_c[r] for r in edit_rank_c]
    edit_id_c = [edit_id_c[r] for r in edit_rank_c]
    edit_proba_c = [edit_proba_c[r] for r in edit_rank_c]
    
    return edit_type_c, edit_id_c, edit_proba_c
    
def get_bg_partition(bg):
    sg = bg.remove_self_loop()
    gs = dgl.unbatch(sg)
    nodes_sep = [0]
    edges_sep = [0]
    for g in gs:
        nodes_sep.append(nodes_sep[-1] + g.num_nodes())
        edges_sep.append(edges_sep[-1] + g.num_edges())
    return gs, nodes_sep[1:], edges_sep[1:]

def write_edits(args, model, test_loader):
    model.eval()
    with open(args['result_path'], 'w') as f:
        f.write('Test_id\tProduct\t%s\n' % '\t'.join(['Prediction %s' % (i+1) for i in range(args['top_num'])]))
        with torch.no_grad():
            for batch_id, data in enumerate(test_loader):
                smiles_list, bg, rxns = data
                batch_atom_logits, batch_bond_logits, _ = predict(args, model, bg)    
                sg = bg.remove_self_loop()
                graphs = dgl.unbatch(sg)
                batch_atom_logits = nn.Softmax(dim = 1)(batch_atom_logits)
                batch_bond_logits = nn.Softmax(dim = 1)(batch_bond_logits) 
                graphs, nodes_sep, edges_sep = get_bg_partition(bg)
                start_node = 0
                start_edge = 0
                print('\rWriting test molecule batch %s/%s' % (batch_id, len(test_loader)), end='', flush=True)
                for single_id, (graph, end_node, end_edge) in enumerate(zip(graphs, nodes_sep, edges_sep)):
                    smiles = smiles_list[single_id]
                    test_id = (batch_id * args['batch_size']) + single_id
                    pred_types, pred_sites, pred_scores = combined_edit(graph, batch_atom_logits[start_node:end_node], batch_bond_logits[start_edge:end_edge], args['top_num'])
                    start_node = end_node
                    start_edge = end_edge
                    f.write('%s\t%s\t%s\n' % (test_id, smiles, '\t'.join(['(%s, %s, %s, %.3f)' % (pred_types[i], pred_sites[i][0], pred_sites[i][1], pred_scores[i]) for i in range(args['top_num'])])))

    print ()
    return 