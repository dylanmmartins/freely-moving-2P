import numpy as np
import matplotlib.pyplot as plt
import fm2p
import json


with open('/home/dylan/Storage/freely_moving_data/_V1PPC/mcmc_spike_times.json', 'r') as file:
    data = json.load(file)

spike_times = data['DMM056']['pos13']