"""Repository paths. Results go to ./results unless STEMARM_RESULTS_DIR is set."""
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_DIR, "model")
DATA_DIR = os.path.join(REPO_DIR, "data")
RESULTS_DIR = os.path.abspath(os.environ.get("STEMARM_RESULTS_DIR") or os.path.join(REPO_DIR, "results"))
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def result_dir(*parts):
    """Create and return a folder below the results directory."""
    path = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def figure_path(name):
    """Path of a paper figure (without extension) in results/figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)
