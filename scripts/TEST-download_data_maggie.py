#### Pre-amble #####
# Maggie's test file - downloading ANES data 
# and cleaning it to get all demographics we will have to examine
####################

# Libraries
import pandas as pd
import numpy as np

# Downloaded 2024 ANES time series data and codebook
raw.data = pd.read.csv("data/anes_timeseries_2024.csv")

# Codebook - only extract post-election questions pertaining to senate elections
  # We care about how people actually voted, so pre-election results aren't needed
  
# Extract questions related to age, gender, region, education, race, party affiliation

# Once extracted,
