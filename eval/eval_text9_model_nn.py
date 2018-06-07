"""
    Copyright (c) 2018-present. Ben Athiwaratkun
    All rights reserved.
    
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree.
"""
# A sample script for qualitative evaluation by querying nearest neighbors
import sys, os
sys.path.append("../")
sys.path.append(".")
import embeval
import numpy as np
import pickle
## First, let's look at the nearest neighbors

basename='modelfiles/multi_text9_e10_d300_vs2e-4_lr1e-5_margin1'
print "Basename = ", basename

ft = embeval.get_fts([basename], multi=True)[0]

for word in ['rock', 'star', 'cell', 'left']:
  print "Nearest Neighbors for {}, cluster 0".format(word)
  print ft.show_nearest_neighbors(word, cl=0)
  print "Nearest Neighbors for {}, cluster 1".format(word)
  print ft.show_nearest_neighbors(word, cl=1)