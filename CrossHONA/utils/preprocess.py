import scanpy as sc
import numpy as np

def fix_var_index_conflict(adata):
    if adata.var.index.name in adata.var.columns:
        if not (adata.var.index.to_series().equals(adata.var[adata.var.index.name])):
            adata.var.index.name = None
        else:
            adata.var = adata.var.drop(columns=[adata.var.index.name])

def _concat_ref_or_target_all(adata_homo, adata_nonhomo):
    adata_all = sc.concat([adata_homo, adata_nonhomo], axis=1, join="outer", merge="same")
    return adata_all

def _cap_to_top_n_within_subset(adata_subset, hvg_rank_series, n):
    if n is None:
        return adata_subset
    if adata_subset.n_vars <= n:
        return adata_subset

    ranks = hvg_rank_series.reindex(adata_subset.var_names)
    ranks = ranks.dropna().sort_values()
    keep = ranks.index[:n].to_numpy()
    return adata_subset[:, keep]

def preprocess(args, adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo, valid_pairs,
               save_data=True):
    print('Reference Homologous Gene:', adata_ref_homo.var.shape[0])
    print('Target Homologous Gene:', adata_target_homo.var.shape[0])
    print('Reference Non-Homologous Gene:', adata_ref_nonhomo.var.shape[0])
    print('Target Non-Homologous Gene:', adata_target_nonhomo.var.shape[0])

    adata_ref_homo.write_h5ad(f'{args.savedir}/adata_ref_homo.h5ad')
    adata_ref_nonhomo.write_h5ad(f'{args.savedir}/adata_ref_nonhomo.h5ad')
    adata_target_homo.write_h5ad(f'{args.savedir}/adata_target_homo.h5ad')
    adata_target_nonhomo.write_h5ad(f'{args.savedir}/adata_target_nonhomo.h5ad')

    #====== homo+nonhomo selcet top args.hvg HVG ======
    adata_ref_all = _concat_ref_or_target_all(adata_ref_homo, adata_ref_nonhomo)
    adata_target_all = _concat_ref_or_target_all(adata_target_homo, adata_target_nonhomo)

    sc.pp.highly_variable_genes(adata_ref_all, flavor='seurat_v3', n_top_genes=int(args.hvg))
    sc.pp.highly_variable_genes(adata_target_all, flavor='seurat_v3', n_top_genes=int(args.hvg))

    hvg_set_ref_all = set(adata_ref_all.var.index[adata_ref_all.var['highly_variable']])
    hvg_set_target_all = set(adata_target_all.var.index[adata_target_all.var['highly_variable']])

    # rank（use for nonhomo cut
    ref_rank = adata_ref_all.var.get('highly_variable_rank', None)
    tgt_rank = adata_target_all.var.get('highly_variable_rank', None)

    # homo: use valid_pairs in full HVG set
    hvg_intersect_ref = []
    hvg_intersect_target = []
    for ref_gene, target_gene in valid_pairs:
        if (ref_gene in hvg_set_ref_all) and (target_gene in hvg_set_target_all) \
           and (ref_gene in adata_ref_homo.var_names) and (target_gene in adata_target_homo.var_names):
            hvg_intersect_ref.append(ref_gene)
            hvg_intersect_target.append(target_gene)

    hvg_intersect_ref = np.array(hvg_intersect_ref)
    hvg_intersect_target = np.array(hvg_intersect_target)

    adata_ref_homo = adata_ref_homo[:, hvg_intersect_ref]
    adata_target_homo = adata_target_homo[:, hvg_intersect_target]
    print(f"Number of HVG (global) that are homo-paired in ref_homo is {len(hvg_intersect_ref)}.", flush=True)
    print(f"Number of HVG (global) that are homo-paired in target_homo is {len(hvg_intersect_target)}.", flush=True)

    # nonhomo: select nonhomo from HVG sets
    sc.pp.filter_genes(adata_target_nonhomo, min_cells=5)

    ref_nonhomo_keep = np.array([g for g in adata_ref_nonhomo.var_names if g in hvg_set_ref_all])
    tgt_nonhomo_keep = np.array([g for g in adata_target_nonhomo.var_names if g in hvg_set_target_all])

    adata_ref_nonhomo = adata_ref_nonhomo[:, ref_nonhomo_keep]
    adata_target_nonhomo = adata_target_nonhomo[:, tgt_nonhomo_keep]

    # nonhomo limit args.nonhomo_hvg genes
    if getattr(args, "nonhomo_hvg", None) is not None:
        if ref_rank is not None:
            adata_ref_nonhomo = _cap_to_top_n_within_subset(adata_ref_nonhomo, ref_rank, args.nonhomo_hvg)
        if tgt_rank is not None:
            adata_target_nonhomo = _cap_to_top_n_within_subset(adata_target_nonhomo, tgt_rank, args.nonhomo_hvg)

    print(f"Number of nonhomo genes kept in ref_nonhomo after global HVG filter: {adata_ref_nonhomo.n_vars}", flush=True)
    print(f"Number of nonhomo genes kept in tgt_nonhomo after global HVG filter: {adata_target_nonhomo.n_vars}", flush=True)

    # ====== save ======
    if save_data:
        fix_var_index_conflict(adata_ref_homo)
        fix_var_index_conflict(adata_ref_nonhomo)
        fix_var_index_conflict(adata_target_homo)
        fix_var_index_conflict(adata_target_nonhomo)

        adata_ref_homo.write_h5ad(f'{args.savedir}/adata_ref_homo_preprocessed.h5ad')
        adata_ref_nonhomo.write_h5ad(f'{args.savedir}/adata_ref_nonhomo_preprocessed.h5ad')
        adata_target_homo.write_h5ad(f'{args.savedir}/adata_target_homo_preprocessed.h5ad')
        adata_target_nonhomo.write_h5ad(f'{args.savedir}/adata_target_nonhomo_preprocessed.h5ad')

    return adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo

def load_preprocessed(savedir):
    adata_ref_homo = sc.read_h5ad(f'{savedir}/adata_ref_homo_preprocessed.h5ad')
    adata_ref_nonhomo = sc.read_h5ad(f'{savedir}/adata_ref_nonhomo_preprocessed.h5ad')
    adata_target_homo = sc.read_h5ad(f'{savedir}/adata_target_homo_preprocessed.h5ad')
    adata_target_nonhomo = sc.read_h5ad(f'{savedir}/adata_target_nonhomo_preprocessed.h5ad')
    return adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo
