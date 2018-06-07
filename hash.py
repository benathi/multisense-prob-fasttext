"""
    Copyright (c) 2018-present. Ben Athiwaratkun
    All rights reserved.
    
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree.
"""
import subprocess
import numpy as np

# BenA:
# the constant values here reflect the values in original FastText implementation

BOW = "<"
EOW = ">"

M32 = 0xffffffffL

def m32(n):
  return n & M32

def mmul(a, b):
  return m32(a*b)

def hash(str):
  h = m32(2166136261L)
  for c in str:
    cc = m32(long(ord(c)))
    h = m32(h ^ cc)
    h = mmul(h, 16777619L)
  return h