"""
Replot ROC curves -- English only, font-cache reset included.
Reads existing test_probs.npy / test_lbls.npy; no retraining needed.
"""

# ── Step 0: clear matplotlib font cache BEFORE importing pyplot ──────────────
import matplotlib
import shutil, os
cache_dir = matplotlib.get_cachedir()
for f in os.listdir(cache_dir):
    if f.endswith('.json') or f.startswith('fontlist'):
        try:
            os.remove(os.path.join(cache_dir, f))
        except Exception:
            pass
print(f"[Font] Cache cleared: {cache_dir}")

# ── Now safe to import pyplot ─────────────────────────────────────────────────
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Rebuild font list and force DejaVu Sans (always bundled with matplotlib)
fm._load_fontmanager(try_read_cache=False)

# Locate the actual DejaVu Sans .ttf shipped with matplotlib
_dejavu_path = None
for _fp in fm.findSystemFonts(fontpaths=None):
    if 'DejaVuSans.ttf' in _fp or 'DejaVuSans-Regular' in _fp:
        _dejavu_path = _fp
        break
if _dejavu_path is None:
    # Fallback: find any DejaVu file in matplotlib's data dir
    import matplotlib as _mpl
    _mpl_data = os.path.join(os.path.dirname(_mpl.__file__), 'mpl-data', 'fonts', 'ttf')
    for _fn in os.listdir(_mpl_data):
        if 'DejaVuSans' in _fn and _fn.endswith('.ttf') and 'Bold' not in _fn and 'Oblique' not in _fn:
            _dejavu_path = os.path.join(_mpl_data, _fn)
            break

if _dejavu_path:
    print(f"[Font] Using DejaVu Sans from: {_dejavu_path}")
    fm.fontManager.addfont(_dejavu_path)

matplotlib.rcParams['font.family']        = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif']    = ['DejaVu Sans', 'Helvetica', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Verify font is active ────────────────────────────────────────────────────
_test_fig, _test_ax = plt.subplots(figsize=(2, 1))
_test_ax.set_title('Font test: AUC = 0.8712 +/- 0.0168')
_test_fig.savefig('/tmp/font_test.png', dpi=72)
plt.close(_test_fig)
print("[Font] Test render OK -> /tmp/font_test.png")

# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# 1. Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR   = "/home/wgf_v100/srtp/\u53e3\u8154\u764c\u5206\u7c7b\u8bc6\u522b\u9879\u76ee"
SAVE_ROOT  = os.path.join(BASE_DIR, "results_comparison_ablation_cv5_seed42")
OUTPUT_DIR = os.path.join(SAVE_ROOT, "summary", "roc_replot")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 2. Experiment definitions
# ──────────────────────────────────────────────────────────────
BACKBONE_EXPS = {
    'Backbone_ResNet50':    ('ResNet-50',               '#1f77b4', '-',  1.8),
    'Backbone_ConvNeXt':    ('ConvNeXt-Tiny',           '#2ca02c', '-',  1.8),
    'Backbone_EffB3':       ('EfficientNet-B3',         '#9467bd', '-',  1.8),
    'Backbone_MobileNetV3': ('MobileNet-V3-L',          '#e377c2', '-',  1.8),
    'Backbone_DenseNet121': ('DenseNet-121',             '#17becf', '-',  1.8),
    'Ref_ImgSE':            ('EfficientNetV2-S',        'black',   '--', 2.5),
}

ABLATION_EXPS = {
    'Ablation_Unimodal':  ('Unimodal (Image Only)',            '#d62728', '-',  1.8),
    'Meta_Only':          ('Unimodal (Metadata Only)',         '#1a9850', '-',  1.8),
    'Ablation_NoSE':      ('No SE-Block',                      '#ff7f0e', '-',  1.8),
    'Ablation_NoEMA':     ('No EMA',                           '#8c564b', '-',  1.8),
    'Ablation_WCE':       ('Weighted CE (no label smoothing)', '#7f7f7f', '-',  1.8),
    'Ref_ImgSE':          ('EfficientNetV2-S Full Model',      'black',   '--', 2.5),
}

# Meta_Only lives in a different results folder
META_SAVE_ROOT = os.path.join(BASE_DIR,
                              "results_meta_only_cv5_seed42_optimized")

# ──────────────────────────────────────────────────────────────
# 3. Data loader
# ──────────────────────────────────────────────────────────────
def load_folds(exp_name, n_folds=5):
    variants = [exp_name]
    if 'ImgSE' in exp_name or 'imgse' in exp_name.lower():
        variants += ['Ref_ImgSE', 'Ref_imgse', 'RefImgSE', 'Ablation_ImgSE']

    # Meta_Only uses a different root directory
    roots = [META_SAVE_ROOT] if exp_name == 'Meta_Only' else []
    roots.append(SAVE_ROOT)

    for root in roots:
        for variant in variants:
            candidate = []
            for fold in range(1, n_folds + 1):
                fold_dir   = os.path.join(root, variant, f"fold_{fold}")
                # Meta_Only folds may sit directly under root/fold_N/
                if not os.path.isdir(fold_dir) and root == META_SAVE_ROOT:
                    fold_dir = os.path.join(root, f"fold_{fold}")
                probs_path = os.path.join(fold_dir, "test_probs.npy")
                lbls_path  = os.path.join(fold_dir, "test_lbls.npy")
                if os.path.exists(probs_path) and os.path.exists(lbls_path):
                    candidate.append({'probs': np.load(probs_path),
                                       'lbls':  np.load(lbls_path)})
            if candidate:
                tag = f" (root: .../{Path(root).name}, as '{variant}')" \
                      if variant != exp_name or root != SAVE_ROOT else ""
                print(f"  [OK]   '{exp_name}'{tag}  --  {len(candidate)} folds")
                return candidate

    existing = sorted([d.name for d in Path(SAVE_ROOT).iterdir() if d.is_dir()]) \
               if Path(SAVE_ROOT).exists() else ['<missing>']
    print(f"  [MISS] '{exp_name}'  --  tried {variants}")
    print(f"         existing: {existing}")
    return []

# ──────────────────────────────────────────────────────────────
# 4. Plot
# ──────────────────────────────────────────────────────────────
def plot_roc_group(exp_dict, title, save_path):
    mean_fpr = np.linspace(0, 1, 300)
    fig, ax  = plt.subplots(figsize=(9, 8))

    loaded_any = False
    for exp_name, (label, color, ls, lw) in exp_dict.items():
        fold_data = load_folds(exp_name)
        if not fold_data:
            continue

        tprs, aucs = [], []
        for r in fold_data:
            fpr, tpr, _ = roc_curve(r['lbls'], r['probs'][:, 1])
            aucs.append(auc(fpr, tpr))
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        mean_tpr     = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc, std_auc = np.mean(aucs), np.std(aucs)

        if ls != '--':
            std_tpr = np.std(tprs, axis=0)
            ax.fill_between(mean_fpr,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            alpha=0.10, color=color)

        ax.plot(mean_fpr, mean_tpr, color=color, linewidth=lw, linestyle=ls,
                label=f'{label}   AUC = {mean_auc:.4f} +/- {std_auc:.4f}')
        print(f"         AUC = {mean_auc:.4f} +/- {std_auc:.4f}")
        loaded_any = True

    if not loaded_any:
        plt.close(); return

    ax.plot([0,1],[0,1], color='#aaaaaa', linewidth=1, linestyle=':',
            label='Random Classifier (AUC = 0.5000)')

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate',  fontsize=13)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.92, edgecolor='#cccccc')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}\n")

# ──────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────
print("\n[1/2] Backbone comparison ...")
plot_roc_group(
    BACKBONE_EXPS,
    title     = ('Receiver Operating Characteristic (ROC) Curves\n'
                 'Backbone Architecture Comparison  (5-Fold Cross-Validation)'),
    save_path = os.path.join(OUTPUT_DIR, 'roc_backbone_replot.png'),
)

print("[2/2] Ablation study ...")
plot_roc_group(
    ABLATION_EXPS,
    title     = ('Receiver Operating Characteristic (ROC) Curves\n'
                 'Ablation Study on Model Components  (5-Fold Cross-Validation)'),
    save_path = os.path.join(OUTPUT_DIR, 'roc_ablation_replot.png'),
)

print(f"\nDone. -> {OUTPUT_DIR}")