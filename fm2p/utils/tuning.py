import numpy as np
import scipy.stats


def tuning_curve(sps, x, x_range):
    """ Calculate tuning curve of neurons to a 1D variable.

    Parameters
    ----------
    sps : np.array
        Spike data. Shape should be (n_cells, n_timepoints).
    x : np.array
        Variable data. Shape should be (n_cells, n_timepoints). The
        timepoints should match those for `sps`, either by interpolation
        or by binning.
    x_range : np.array
        Array of values to bin x into.
    
    Returns
    -------
    var_cent : np.array
        Array of values at the center of each bin. Shape is (n_bins,)
    tuning : np.array
        Array of mean spike counts for each bin. Shape is (n_cells, n_bins).
    tuning_err : np.array
        Array of standard error of the mean spike counts for each bin. Shape
        is (n_cells, n_bins).
    """

    n_cells = np.size(sps,0)

    scatter = np.zeros((n_cells, np.size(x,0)))

    tuning = np.zeros((n_cells, len(x_range)-1))
    tuning_err = tuning.copy()
    var_cent = np.zeros(len(x_range)-1)
    
    for j in range(len(x_range)-1):
        
        var_cent[j] = 0.5*(x_range[j] + x_range[j+1])
    
    for n in range(n_cells):
        
        scatter[n,:] = sps[n,:]
        
        for j in range(len(x_range)-1):
            
            usePts = (x>=x_range[j]) & (x<x_range[j+1])
            
            tuning[n,j] = np.nanmean(scatter[n, usePts])
            
            tuning_err[n,j] = np.nanstd(scatter[n, usePts]) / np.sqrt(np.count_nonzero(usePts))

    return var_cent, tuning, tuning_err


def plot_tuning(ax, var_cent, tuning, tuning_err, color, rad=True):
    if rad:
        usebins = np.rad2deg(var_cent)
    else:
        usebins = var_cent.copy()
    ax.plot(usebins, tuning[0], color=color)
    ax.fill_between(
        usebins,
        tuning[0]+tuning_err[0],
        tuning[0]-tuning_err[0],
        alpha=0.3, color=color
    )
    ax.set_xlim([var_cent[0], var_cent[-1]])


def calc_modind(bins, tuning, fr, thresh=0.33):
    # modind of 0.33 is a doubling of firing rate relative

    # mean firing rate across the recording
    b = np.nanmean(fr)
    peak_val = np.nanmax(tuning)

    # print(b, peak_val)

    # diff over sum
    modind = (peak_val - b) / (peak_val + b)

    peak = np.nan
    if modind > 0.33:
        peak = bins[np.nanargmax(tuning)]

    return modind, peak



def calc_tuning_reliability(spikes, behavior, bins, splits_inds):
            
    cnk_mins = []
    cnk_maxs = []

    for cnk in range(len(splits_inds)):
        hist_cents, cnk_behavior_tuning, _ = tuning_curve(
            spikes[np.newaxis, splits_inds[cnk]],
            behavior[splits_inds[cnk]],
            bins
        )
        cnk_mins = hist_cents[np.nanargmin(cnk_behavior_tuning)]
        cnk_maxs = hist_cents[np.nanargmax(cnk_behavior_tuning)]

    try:
        pval_across_cnks = scipy.stats.wilcoxon(
            cnk_mins,
            cnk_maxs,
            alternative='less'
        ).pvalue
    except ValueError:
        print('x-y==0 for all elements of this cell, which cannot be computed for wilcox. Skipping this cell.')
        pval_across_cnks = np.nan

    # If the p value is small, the two distributions are significantly different from
    # one another, i.e., the peaks are all different from the troughs. This means that
    # the cell has a reliable peak.

    return pval_across_cnks


