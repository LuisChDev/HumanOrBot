import os
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import xml.etree.ElementTree as xml
# from sklearn.pipeline import make_pipeline

# importing the dataset
# directorios
dirdata = 'pan19_author_profiling_training_es/'
dirname = os.path.abspath('.')

users = []
for file_ in ["".join([dirname, '/', dirdata, x]) for x in os.listdir(
        os.path.join(dirname, dirdata))]:
    xml.parse(file_)
    # print(file_)
