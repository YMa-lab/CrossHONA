"""
Plot confusion matrix for the LogReg classifier on the homo_mean latent of
the Merfish_NonHomo run, on both ref and target.

LogReg is fit on ref homo_mean -> ref_y, then predicts ref and target.
Outputs PNG confusion matrices in the run directory.
"""
import os
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

RUN_DIR = "resultdir/Merfish_NonHomo"
PARENT  = "resultdir"

# ---------- load latent + labels ----------
X_ref = np.load(os.path.join(RUN_DIR, "ref_homo_mean.npy"))
X_tgt = np.load(os.path.join(RUN_DIR, "tgt_homo_mean.npy"))
y_ref = np.load(os.path.join(RUN_DIR, "ref_gt.npy")).astype(int)
y_tgt = np.load(os.path.join(RUN_DIR, "target_gt.npy")).astype(int)

print(f"ref: {X_ref.shape}, classes {sorted(np.unique(y_ref))}")
print(f"tgt: {X_tgt.shape}, classes {sorted(np.unique(y_tgt))}")

# ---------- inverse label dicts (int -> name) ----------
with open(os.path.join(PARENT, "inverse_dict_ref.pkl"), "rb") as f:
    inv_ref = pickle.load(f)
with open(os.path.join(PARENT, "inverse_dict_target.pkl"), "rb") as f:
    inv_tgt = pickle.load(f)

# Build a unified id -> name lookup. Prefer ref names; fall back to tgt.
all_ids = sorted(set(np.concatenate([np.unique(y_ref), np.unique(y_tgt)])))
id2name = {}
for i in all_ids:
    if i in inv_ref:
        id2name[i] = inv_ref[i]
    elif i in inv_tgt:
        id2name[i] = inv_tgt[i]
    else:
        id2name[i] = f"cls_{i}"
print("Cell type lookup:", id2name)

# ---------- LogReg ----------
clf = LogisticRegression(max_iter=1000, n_jobs=-1)
clf.fit(X_ref, y_ref)
preds_ref = clf.predict(X_ref)
preds_tgt = clf.predict(X_tgt)
print(f"ref acc = {(preds_ref == y_ref).mean():.4f}")
print(f"tgt acc = {(preds_tgt == y_tgt).mean():.4f}")

# ---------- confusion matrices ----------
def plot_cm(y_true, y_pred, title, save_path,
            label_ids, id2name, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm.astype(float)

    n = len(label_ids)
    names = [id2name.get(i, str(i)) for i in label_ids]

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * n + 2),
                                    max(5, 0.45 * n + 2)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1 if normalize else cm_norm.max())
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)

    # annotate
    thresh = cm_norm.max() * 0.5
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if normalize:
                txt = f"{v:.2f}" if v > 0.005 else ""
            else:
                txt = f"{int(v)}" if v > 0 else ""
            if txt:
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if v > thresh else "black", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {save_path}")

label_ids = sorted(id2name.keys())

plot_cm(y_ref, preds_ref,
        title=f"LogReg confusion matrix (ref)  acc={(preds_ref == y_ref).mean():.3f}",
        save_path=os.path.join(RUN_DIR, "logreg_confusion_ref.png"),
        label_ids=label_ids, id2name=id2name, normalize=True)

plot_cm(y_tgt, preds_tgt,
        title=f"LogReg confusion matrix (tgt)  acc={(preds_tgt == y_tgt).mean():.3f}",
        save_path=os.path.join(RUN_DIR, "logreg_confusion_tgt.png"),
        label_ids=label_ids, id2name=id2name, normalize=True)

# also raw counts (un-normalised) for the target — easier to spot which
# tiny classes get totally missed
plot_cm(y_tgt, preds_tgt,
        title=f"LogReg confusion matrix (tgt, counts)",
        save_path=os.path.join(RUN_DIR, "logreg_confusion_tgt_counts.png"),
        label_ids=label_ids, id2name=id2name, normalize=False)

print("done.")
