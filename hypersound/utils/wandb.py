import io

import wandb
from matplotlib.figure import Figure
from PIL import Image

# isort: split


# ----------------------------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------------------------
def fig_to_wandb(fig: Figure) -> wandb.Image:
    """Convert a matplotlib.Figure to a wandb.Image.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure.

    Returns
    -------
    wandb.Image
        Image version of the figure.

    """
    buffer = io.BytesIO()
    fig.savefig(buffer, bbox_inches="tight")  # type: ignore

    with Image.open(buffer) as img:
        img = wandb.Image(img)

    return img
