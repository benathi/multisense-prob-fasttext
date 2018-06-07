"""
    Copyright (c) 2018-present. Ben Athiwaratkun
    All rights reserved.
    
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree.
"""
# A sample script for quantitative evaluation on word similarity datasets
import sys, os
sys.path.append("../")
sys.path.append(".")
import embeval
import numpy as np
import pickle
import argparse

def calculate_wordsim(modelname, multi=1, verbose=True, name='model'):
  if verbose: print "Basename to evaluate =", modelname
  multift = embeval.get_fts([modelname], multi=multi)[0]
  if multi:
    df = embeval.eval_multi(multift, lower=False)
    if verbose:
      print "Result DataFrame"
      print df
    return df
  else:
    df = embeval.eval_ft(multift, name=name)
    if verbose:
      print "Result DataFrame"
      print df
    return df

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--modelname', default='', type=str, help="The model to be evaluated. For instance, the files 'modelname.in', 'modelname.bin', etc should exist.")
  parser.add_argument('--multi', default=1, type=int, help="Whether this is a multisense model")
  args = parser.parse_args()
  result = calculate_wordsim(modelname=args.modelname,
    multi=args.multi, name=args.modelname)