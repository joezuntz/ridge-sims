import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def plot_segmented_filaments(filament_file, output_path=None, title=None, dpi=300):
    """
    Plot segmented filaments color-coded by filament label.
    """
    with h5py.File(filament_file, 'r') as f:
        data = f['data'][:]
        ra = data['RA']
        dec = data['DEC']
        labels = data['Filament_Label']

    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    cmap = cm.get_cmap('tab20', num_labels)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(ra[mask], dec[mask], s=2, color=cmap(i), label=f'Label {label}' if num_labels <= 10 else None)

    plt.xlabel("RA ")
    plt.ylabel("DEC ")
    plt.title(title or f"Segmented Filaments: {os.path.basename(filament_file)}")
    if num_labels <= 10:
        plt.legend(markerscale=3)
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    filament_file = "example/filaments/filaments_p50.h5"  
    output_plot = "plots/filaments_p50_color.png"
    plot_segmented_filaments(filament_file, output_path=output_plot)
