
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fm2p

def make_standard_diagnostic_pages(data, savepath):
    raise NotImplemented




def make_ltdk_diagnostic_pages(data, savepath):

    pdf = PdfPages(savepath)

    ### Eye tracking








def run_preprocessing_diagnostics(data=None, savepath=None, ltdk=False):

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preproc', type=str, default=None)
    parser.add_argument('-s', '--saveas', type=str, default=None)
    parser.add_argument('-s', '--saveas', type=fm2p.str_to_bool, default=None)
    args = parser.parse_args()


    if not ltdk:
        make_standard_diagnostic_pages(data, savepath)

    elif ltdk:
        make_ltdk_diagnostic_pages(data, savepath)


if __name__ == '__main__':

    run_preprocessing_diagnostics()