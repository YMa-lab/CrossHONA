import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scanpy as sc
import pandas as pd

def plot_umap(adata, key, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(
        adata,
        color=key,
        frameon=False,
        palette=['#432371', '#faae7b'] if key == 'species' else None,
        ax=ax,
        show=False
    )
    ax.legend(title='Dataset', loc='upper left', bbox_to_anchor=(1.01, 1.0))
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_umap_three(a1, k1, a2, k2, a3, k3, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    title_map = {
        k1: "Species ",
        k2: "GT Cell Type",
        k3: "Pred Cell Type"
    }

    for ax, ad, k in zip(axes, [a1, a2, a3], [k1, k2, k3]):
        sc.pl.umap(
            ad,
            color=k,
            frameon=False,
            palette=['#432371', '#faae7b'] if k == 'species' else None,
            ax=ax,
            show=False
        )
        ax.set_title(title_map[k], fontsize=13)
        
        leg = ax.get_legend()
        if leg is not None:
            handles = leg.legend_handles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()

            ax.legend(
                handles,
                labels,
                ncol=1,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=8,
                frameon=False,
            )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


