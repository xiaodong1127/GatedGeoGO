import argparse
import json
import os
import time

from predgo.model import PredGOModel
from tools.tools import print_args_params, map_annotation, load_go_graph


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--binary_dir', type=str,
                        default="data/2016-MMseqs/binary_dir")
    parser.add_argument('-od', '--out_dir', default='data/2016-MMseqs/result', type=str)

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
                        default="cpu", help="device.")

    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    print_args_params(args)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('device:', args.device)
    terms_embed = map_annotation(args.term_file_path, args.ont)
    index2term = {v: k for k, v in terms_embed.items()}
    num_class = len(terms_embed)
    go_graph = load_go_graph(args.go_graph_path)
    model = PredGOModel()

    model.init_model(num_class=num_class, aa_node_in_dim=(6, 3),
                     aa_node_h_dim=(24, 12),
                     aa_edge_in_dim=(32, 1), aa_edge_h_dim=(128, 4), gvp_out_dim=12, num_gvp_layers=2,
                     ppn_num_heads=8,
                     num_ppn_layers=2, drop_rate=0.2)
    model.load_specified_model(args.model_path, device=args.device, need_train=False)
    score_list, files = model.predict_from_binary_dir(args.binary_dir, args.device)
    for score, file in zip(score_list, files):
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
        print(go_scores)
        with open(os.path.join(args.out_dir, f'{file[0]}_go_score.json'), 'w') as f:
            f.write(json.dumps(go_scores))
