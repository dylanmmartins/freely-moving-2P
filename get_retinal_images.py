import numpy as np
import matplotlib.pyplot as plt
from utils.vfs_contours import get_trace, plotVisualCoverage
from utils.paths import choose_most_recent

def generate_diagnostic_figure():
    # Randomly select positions for the pillar and mouse
    pillar_position = (np.random.rand() * 100, np.random.rand() * 100)
    mouse_position = (np.random.rand() * 100, np.random.rand() * 100)
    mouse_orientation = np.random.rand() * 360

    # Load the topdown camera video
    topdown_video_path = choose_most_recent(["path/to/topdown/video"])
    topdown_video = load_data(topdown_video_path)

    # Randomly select a frame from the topdown video
    random_frame_index = np.random.randint(0, len(topdown_video))
    topdown_frame = topdown_video[random_frame_index]

    # Run retinal reconstruction code to get pillar orientations
    pillar_orientations = reconstruct_retina(pillar_position)

    # Generate the elongated version of the retinal image
    elongated_retinal_image = generate_elongated_retinal_image(topdown_frame, pillar_orientations)

    # Create the figure with three panels
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # Top panel: visualize pillar and mouse
    axes[0].imshow(topdown_frame)
    axes[0].plot(pillar_position[0], pillar_position[1], 'ro', markersize=20)
    axes[0].plot([pillar_position[0], mouse_position[0]], [pillar_position[1], mouse_position[1]], 'y-', linewidth=2)
    axes[0].set_xlim(0, topdown_frame.shape[1])
    axes[0].set_ylim(topdown_frame.shape[0], 0)
    axes[0].set_title('Top Panel: Pillar and Mouse')

    # Middle panel: retinal image for a single frame
    axes[1].imshow(elongated_retinal_image)
    axes[1].plot(pillar_position[0], pillar_position[1], 'ro', markersize=20)
    axes[1].set_xlim(0, elongated_retinal_image.shape[1])
    axes[1].set_ylim(elongated_retinal_image.shape[0], 0)
    axes[1].set_title('Middle Panel: Retinal Image')

    # Bottom panel: elongated retinal image with dashed lines at retina borders
    axes[2].imshow(elongated_retinal_image)
    axes[2].plot(pillar_position[0], pillar_position[1], 'ro', markersize=20)
    axes[2].hlines([0, elongated_retinal_image.shape[0]], 0, elongated_retinal_image.shape[1], colors='k', linestyles='dashed')
    axes[2].vlines([0, elongated_retinal_image.shape[1]], 0, elongated_retinal_image.shape[0], colors='k', linestyles='dashed')
    axes[2].set_xlim(0, elongated_retinal_image.shape[1])
    axes[2].set_ylim(elongated_retinal_image.shape[0], 0)
    axes[2].set_title('Bottom Panel: Elongated Retinal Image')

    plt.tight_layout()
    return fig

def reconstruct_retina(pillar_position):
    # Placeholder function for retinal reconstruction
    return np.random.rand(10, 2)

def generate_elongated_retinal_image(retinal_image, pillar_orientations):
    # Placeholder function to generate elongated retinal image
    return np.pad(retinal_image, ((0, 50), (0, 50)), mode='constant')

# Generate ten pages of diagnostic figures
for _ in range(10):
    fig = generate_diagnostic_figure()
    plt.show()
