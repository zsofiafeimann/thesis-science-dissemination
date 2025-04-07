# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 22:24:09 2025

@author: Diak
"""

import pandas as pd
import numpy as np


df_unmatched = pd.read_csv("cleaned_data/unmatched_ranked_institutions.csv", sep="|")
print("Unmatched institutions:", len(df_unmatched))
