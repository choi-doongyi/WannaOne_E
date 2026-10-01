from pathlib import Path
import matplotlib.pyplot as plt
from src.config import PLOT_FONT, SHOW_PLOTS

def setup_plot_font():
    plt.rcParams["font.family"] = PLOT_FONT
    plt.rcParams["axes.unicode_minus"] = False

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def finish_plot(save_path: Path, dpi: int = 300):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()
