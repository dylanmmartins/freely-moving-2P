from matplotlib.patches import Ellipse
from scipy.stats import chi2

def plot_confidence_ellipse(x, y, ax, confidence=0.90, **kwargs):
    """
    Plot a confidence ellipse for 2D data.

    Parameters:
    - x, y: 1D arrays of the same length
    - ax: matplotlib Axes object
    - confidence: confidence level (e.g., 0.90)
    - kwargs: passed to Ellipse patch
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)

    # Eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Compute the angle between the x-axis and the largest eigenvector
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Chi-squared quantile for the given confidence level (2 degrees of freedom)
    chi2_val = chi2.ppf(confidence, df=2)
    width, height = 2 * np.sqrt(vals * chi2_val)

    ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)