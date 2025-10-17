import argparse
import json
import os
import time

from predgo.model import PredGOModel
from tools.tools import print_args_params, map_annotation, read_ppi_info, read_fasta_file, load_go_graph
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tools.structure_data_parser import StructureDataParser
from tools.tools import write_fasta_file
from esm.extract import extract_esm_mean
import pathlib
import torch
from predgo.data import generate_PredGOData

def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--struct_dir', type=str,
                        default="data/2016-MMseqs/afdb_dir")
    parser.add_argument('-wd', '--work_dir', default='data/2016-MMseqs/work_dir', type=str)
    parser.add_argument('-od', '--out_dir', default='data/2016-MMseqs/result', type=str)

    parser.add_argument('--ppi_path', type=str,
                        default="data/2016-MMseqs/ppi_score.tsv")
    parser.add_argument('--ppi_seqs', type=str,
                        default="data/2016-MMseqs/ppi_seqs.fasta")

    parser.add_argument('--term_file_path', type=str,
                        default="data/2016-MMseqs/terms-50.tsv",
                        help="annotation_terms_file_path.")

    parser.add_argument('--model_path', type=str,
                        default="trained_models/2016-MMseqs/mf_model.pth",
                        help="annotation_terms_file_path.")
    parser.add_argument('--go_graph_path', type=str, default="data/2016-MMseqs/go.obo",
                        help="go_graph_path.")
    parser.add_argument('--ont', type=str,
                        default="mf", help="mf bp cc type.")
    parser.add_argument('--device', type=str,
                        default="cuda:0", help="device.")

    return parser




    args = create_parser().parse_args()
    print_args_params(args)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('device:', args.device)
    terms_embed = map_annotation(args.term_file_path, args.ont)
    go_graph = load_go_graph(args.go_graph_path)
    num_class = len(terms_embed)

    ppi_info_dict = read_ppi_info(args.ppi_path)
    ppi_seq_dict = {rec.id: str(rec.seq) for rec in read_fasta_file(args.ppi_seqs)}

    struct_dir = args.struct_dir
    files = os.listdir(struct_dir)
    target_seq_out = os.path.join(args.work_dir, 'temp_target_seqs.fa')
    ppi_seq_out = os.path.join(args.work_dir, 'temp_ppi_seqs.fa')
    ppi_score_out = os.path.join(args.work_dir, 'temp_ppi_score.tsv')
    esm_dir_path = os.path.join(args.work_dir, 'esm_dir')
    seq_rec = []
    ppi_rec_dict = {}
    for file in tqdm(files, total=len(files),desc='extract seqs'):
        t_id = pathlib.Path(file).stem
        struct_parser = StructureDataParser(os.path.join(struct_dir, file), 'target', 'pdb')
        seq = ''.join(struct_parser.get_sequence())
        if 16 <= len(seq) <= 1000:
            rec = SeqRecord(Seq(seq),
                            id=t_id,
                            description="")
            seq_rec.append(rec)
            ppi_list = []
            if t_id in ppi_info_dict:
                ppi_list = ppi_info_dict[t_id]
            for i in ppi_list:
                if i not in ppi_rec_dict:
                    rec = SeqRecord(Seq(ppi_seq_dict[i]),
                                    id=i,
                                    description="")
                    ppi_rec_dict[i] = rec
    ppi_recs = [v for k, v in ppi_rec_dict.items()]
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    write_fasta_file(seq_rec, target_seq_out)
    write_fasta_file(ppi_recs, ppi_seq_out)
    if not os.path.exists(esm_dir_path):
        os.makedirs(esm_dir_path)
    extract_esm_mean(target_seq_out, esm_dir_path, args.device)
    extract_esm_mean(ppi_seq_out, esm_dir_path, args.device)

    model = PredGOModel()
    model.init_model(num_class=num_class, aa_node_in_dim=(6, 3),
                     aa_node_h_dim=(24, 12),
                     aa_edge_in_dim=(32, 1), aa_edge_h_dim=(128, 4), gvp_out_dim=12, num_gvp_layers=2,
                     ppn_num_heads=8,
                     num_ppn_layers=2, drop_rate=0.2)
    model.load_specified_model(args.model_path, device=args.device, need_train=False)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    esm_pre_cache = {}
    for file in tqdm(files, total=len(files),desc='predict'):
        t_id = pathlib.Path(file).stem
        struct_parser = StructureDataParser(os.path.join(struct_dir, file), 'target', 'pdb')
        seq = ''.join(struct_parser.get_sequence())
        if not 16 <= len(seq) <= 1000:
            continue
        coords = struct_parser.get_residue_atoms_coords()
        target_esm_pre = torch.load(os.path.join(esm_dir_path,t_id+'.pt'))['mean_representations'][33]
        if t_id in ppi_info_dict:
            ppi_esm_pre = []
            for v in ppi_info_dict[t_id]:
                if v not in esm_pre_cache:
                    esm_pre_cache[v] = torch.load(os.path.join(esm_dir_path,v+'.pt'))['mean_representations'][33]
                ppi_esm_pre.append(esm_pre_cache[v])
            ppi_esm_pre = torch.stack(ppi_esm_pre,dim=0)
            ppi_esm_pre = torch.cat([target_esm_pre.view(1, -1), ppi_esm_pre], dim=0)
        else:
            ppi_esm_pre = target_esm_pre.view(1, -1)
        target = generate_PredGOData(target_esm_pre, ppi_esm_pre, coords)
        score = model.predict_from_predgo_data(target)
        score = score.view(-1)
        go_scores = []
        for term, idx in terms_embed.items():
            s = round(score[idx].item(), 4)
            if s < 0.01:
                continue
            tmp_info = go_graph.nodes[term]
            term_info = {'term': term, 'score': round(s, 4), 'name': tmp_info['name'],
                        'type': tmp_info['namespace']}
            go_scores.append(term_info)
        go_scores.sort(key=lambda x: x['score'], reverse=True)
        #print(go_scores)
        with open(os.path.join(args.out_dir, f'{t_id}_go_score.json'), 'w') as file:
            file.write(json.dumps(go_scores))
