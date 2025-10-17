import argparse
import time

import numpy as np

from predgo.data import PredGODataset
from predgo.model import PredGOModel
from tools.log import log
from tools.tools import print_args_params


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', type=str,
                        default=r'/data8t/zhengrongtao/proteinDataset/cafa3-0/esm_gvp_ppi')

    parser.add_argument('--train_annotation_path', type=str,
                        default=r'/data8t/zhengrongtao/proteinDataset/cafa3-0/train_data_train_standardAA_1000_afdb.tsv')
    parser.add_argument('--train_seq_dir', type=str,
                        default=r'/data8t/zhengrongtao/proteinDataset/cafa3-0/train_data_train_standardAA_1000_afdb.fasta')

    parser.add_argument('--validation_annotation_path', type=str,
                        default=r'/data8t/zhengrongtao/proteinDataset/cafa3-0/train_data_valid_standardAA_1000_afdb.tsv')
    parser.add_argument('--validation_seq_dir', type=str,
                        default=r'/data8t/zhengrongtao/proteinDataset/cafa3-0/train_data_valid_standardAA_1000_afdb.fasta')

    parser.add_argument('--test_annotation_path', type=str,
                        default=r'/data8t/zhengrongtao/proteinDataset/cafa3-0/test_data_standardAA_1000.tsv')
    parser.add_argument('--test_seq_dir', type=str,
                        default=r'/data8t/zhengrongtao/proteinDataset/cafa3-0/test_data_standardAA_1000.fasta')

    parser.add_argument('--term_file_path', type=str,
                        default="/data8t/zhengrongtao/proteinDataset/cafa3-0/terms.tsv",
                        help="annotation_terms_file_path.")
    parser.add_argument('--esm_dir_path', type=str,
                        default="/data8t/zhengrongtao/proteinDataset/cafa3-0/esm", help="esm_dir_path.")
    parser.add_argument('--esm_mean_dir_path', type=str,
                        default="/data8t/zhengrongtao/proteinDataset/cafa3-0/esm-mean", help="esm_dir_path.")
    parser.add_argument('--struct_dir_path', type=str,
                        default="/data8t/zhengrongtao/proteinDataset/afdb_cafa3", help="esm_dir_path.")
    parser.add_argument('--ppi_path', type=str,
                        default="/data8t/zhengrongtao/proteinDataset/cafa3-0/ppi_score.tsv")

    parser.add_argument('--go_graph_path', type=str, default="/data8t/zhengrongtao/proteinDataset/cafa3-0/go.obo",
                        help="go_graph_path.")
    parser.add_argument('--model_path', type=str,
                        default="trained_models/ESMGVPPPN/2016/cc_model.pth",
                        help="go_graph_path.")

    parser.add_argument('--ont', type=str,
                        default="mf", help="mf bp cc type.")
    parser.add_argument('--device', type=str,
                        default="cuda:1", help="device.")
    parser.add_argument('--batch_size', type=int,
                        default="24", help="batch_size.")
    parser.add_argument('--num_epochs', type=int,
                        default="15", help="num_epochs.")
    parser.add_argument("--local_rank", default=0, type=int)

    return parser


def diamond_predict(diamond_file_path, train_df, test_df, ont):
    diamond_scores = {}
    with open(diamond_file_path) as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in diamond_scores:
                diamond_scores[it[0]] = {}
            diamond_scores[it[0]][it[1]] = float(it[-1])

    train_df[ont] = train_df[ont].fillna('')
    annotations = train_df[ont].values
    annotations = list(map(lambda x: set(x.split(',')), annotations))
    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.protein_id] = i
    blast_preds = []
    for i, row in enumerate(test_df.itertuples()):
        annots = {}
        prot_id = row.protein_id
        # BlastKNN
        if prot_id in diamond_scores:
            sim_prots = diamond_scores[prot_id]
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                if p_id in prot_index:
                    allgos |= annotations[prot_index[p_id]]
                    total_score += score
            # allgos 相似的蛋白质中出现的GO
            allgos = list(sorted(allgos))
            sim = np.zeros(len(allgos), dtype=np.float32)
            # 可能有多个蛋白质都有相同的GO 对于每一个出现的GO取最大的bit score
            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if p_id in prot_index:
                        if go_id in annotations[prot_index[p_id]]:
                            s = max(s, score)
                sim[j] = s
            if len(sim) > 0:
                sim = sim / np.max(sim)
            for go_id, score in zip(allgos, sim):
                annots[go_id] = score
        blast_preds.append(annots)
    return blast_preds


if __name__ == '__main__':
    args = create_parser().parse_args()

    print_args_params(args)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('device:', args.device)

    train_dataset = PredGODataset(root=args.root_dir, annotation_path=args.train_annotation_path,
                                  sequences_path=args.train_seq_dir,
                                  terms_file_path=args.term_file_path,
                                  struct_dir_path=args.struct_dir_path,
                                  esm_dir_path=args.esm_dir_path,
                                  esm_mean_dir_path=args.esm_mean_dir_path,
                                  ppi_path=args.ppi_path,
                                  annotation_type=args.ont,
                                  seq_max_len=1000, seq_min_len=16, max_ppi_data_num=200,
                                  data_overwrite=False, skip_load=True)

    val_dataset = PredGODataset(root=args.root_dir, annotation_path=args.validation_annotation_path,
                                sequences_path=args.validation_seq_dir,
                                terms_file_path=args.term_file_path,
                                struct_dir_path=args.struct_dir_path,
                                esm_dir_path=args.esm_dir_path,
                                esm_mean_dir_path=args.esm_mean_dir_path,
                                ppi_path=args.ppi_path,
                                annotation_type=args.ont,
                                seq_max_len=1000, seq_min_len=16, max_ppi_data_num=200,
                                data_overwrite=False, skip_load=True)

    test_dataset = PredGODataset(root=args.root_dir, annotation_path=args.test_annotation_path,
                                 sequences_path=args.test_seq_dir,
                                 terms_file_path=args.term_file_path,
                                 struct_dir_path=args.struct_dir_path,
                                 esm_dir_path=args.esm_dir_path,
                                 esm_mean_dir_path=args.esm_mean_dir_path,
                                 ppi_path=args.ppi_path,
                                 annotation_type=args.ont,
                                 seq_max_len=1000, seq_min_len=16, max_ppi_data_num=200,
                                 data_overwrite=False, skip_load=True)

    log.do_print(f'train_dataset len: {len(train_dataset)}')
    log.do_print(f'val_dataset len: {len(val_dataset)}')
    log.do_print(f'test_dataset len: {len(test_dataset)}')

    num_class = test_dataset.terms_embed_num

    model = PredGOModel()

    model.init_smin_calculator(go_graph_path=args.go_graph_path, train_data=train_dataset, test_data=test_dataset)
    model.init_model(num_class=num_class, aa_node_in_dim=(6, 3),
                     aa_node_h_dim=(24, 12),
                     aa_edge_in_dim=(32, 1), aa_edge_h_dim=(128, 4), gvp_out_dim=12, num_gvp_layers=2,
                     ppn_num_heads=8,
                     num_ppn_layers=2, drop_rate=0.2)

    log.do_print('evaluate test dataset:')
    model.load_specified_model(args.model_path, device=args.device, need_train=False)
    test_result = model.evaluate_dataset(test_dataset, device=args.device)

    log.do_print('evaluate test human dataset:')
    human_dataset = test_dataset.get_sub_dataset_by_orgs(9606)
    log.do_print(f'size: {len(human_dataset)}')
    human_test_result = model.evaluate_dataset(human_dataset, device=args.device)

    log.do_print('evaluate test mouse dataset:')
    mouse_dataset = test_dataset.get_sub_dataset_by_orgs(10090)
    log.do_print(f'size: {len(mouse_dataset)}')
    mouse_test_result = model.evaluate_dataset(mouse_dataset, device=args.device)

    log.do_print('evaluate test Fission yeast dataset:')
    yeast_dataset = test_dataset.get_sub_dataset_by_orgs(284812)
    log.do_print(f'size: {len(yeast_dataset)}')
    _ = model.evaluate_dataset(yeast_dataset, device=args.device)
