import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ROC曲线画图函数
def draw_roc(fpr, tpr, roc_auc, title='ROC curve'):
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(fpr, tpr, 'k--', label='ROC (area = %0.2f)' % roc_auc, lw=2)
    plt.xlim([0.00, 1.00])
    plt.ylim([0.00, 1.00])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('%s' % title, fontsize=18)
    plt.legend(loc='lower right')
    plt.show()


def mean_roc_plot(ax, tprs, aucs, mean_fpr, figname, output_path=None):
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=figname,
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    if output_path is None:
        path = Path("./Figure")
        path.mkdir(parents=True, exist_ok=True)
        output_path = path / f"{figname}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=600)
    # plt.show()


def DrawROC(*args, **kwargs):
    return draw_roc(*args, **kwargs)


def Mean_roc_plot(*args, **kwargs):
    return mean_roc_plot(*args, **kwargs)


__all__ = ["draw_roc", "mean_roc_plot", "DrawROC", "Mean_roc_plot"]
