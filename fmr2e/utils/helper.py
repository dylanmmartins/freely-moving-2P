import pandas as pd
import numpy as np

def split_xyl(xyl):

    names = list(xyl.columns.values)

    x_locs = []
    y_locs = []
    l_locs = []

    # seperate the lists of point names into x, y, and likelihood
    for loc_num in range(0, len(names)):
        loc = names[loc_num]
        if '_x' in loc:
            x_locs.append(loc)
        elif '_y' in loc:
            y_locs.append(loc)
        elif 'likeli' in loc:
            l_locs.append(loc)

    x_vals = xyl[x_locs]
    y_vals = xyl[y_locs]
    l_vals = xyl[l_locs]

    return x_vals, y_vals, l_vals


def apply_liklihood_thresh(x, l):

    thresh_arr = (l>0.99).astype(float).values
    x_vals1 = x.copy().values

    x_vals2 = pd.DataFrame((x_vals1 * thresh_arr), columns=x.columns)
    x_vals2[x_vals2==0.] = np.nan

    x_vals = x_vals2.copy()

    return x_vals