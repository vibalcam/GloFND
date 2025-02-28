import matplotlib.pyplot as plt
import wandb

def log_hist_as_image(data, title, bins=64, dpi=300):
    # Create a figure with high DPI for better quality
    fig = plt.figure(figsize=(6, 4), dpi=dpi)
    plt.hist(data, bins=bins, density=True)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to an image and log with wandb
    wandb.log({title: wandb.Image(fig)}, commit=False)
    plt.close(fig)


def update_dict(d, u, prefix='', sufix=''):
    d.update({prefix + k + sufix: v for k, v in u.items()})