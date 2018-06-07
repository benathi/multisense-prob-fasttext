"""
    Copyright (c) 2018-present. Ben Athiwaratkun
    All rights reserved.
    
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree.
"""
from hash import hash
import numpy as np
import argparse, os
import pandas as pd
import timeit
import scipy

BOW = "<"
EOW = ">"

def computeNgrams(word, minn=3, maxn=6, bucket=2000000):
    ngram_hashes = []
    ngrams = []
    for i, cc in enumerate(word):
      c = ord(cc)
      ngram = ""
      if ((c & 0xC0) == 0x80): continue
      j = i
      n = 1
      while j < len(word) and n <= maxn:
        ngram += word[j]
        j += 1
        while (j < len(word) and ( ord(word[j]) & 0xC0) == 0x80):
          ngram += word[j];
          j += 1
        if n >= minn and not (n == 1 and (i == 0 or j == len(word))):
          h = hash(ngram) % bucket
          ngram_hashes.append(h)
          ngrams.append(ngram)
        n += 1
    return ngram_hashes, ngrams

class MultiFastText:
  def __init__(self, basename, minn=3, maxn=6, bucket=2000000, cache=True, verbose=False,
    dein=True, multi=False, savevec=False):
    if verbose: print("Constructing MultiFt Object: Basename = {}".format(basename))
    self.basename = basename
    self.bucket = bucket
    self.cache = cache
    self.minn = minn
    self.maxn = maxn
    self.multi = multi
    if self.maxn == 0:
      self.bucket = 0
    self.dein = True   # original ft
    self.deout = False # original ft
    if "no-dein" in basename:
      print "Setting dein = False based on the basename"
      self.dein = False

    dict_fname = basename + ".words"
    in_fname = basename + ".in"
    out_fname = basename + ".out"
    if verbose: print "Loading Words from Dictionary"
    self.load_dict(dict_fname)
    if verbose: print "Loading Emb Out"
    self.emb_out = self.load_emb_out(out_fname)
    if verbose: print "Loading Emb In"
    self.emb = self.load_emb_in(in_fname)

    # meaning that for multi-prototype, the second cluster uses vector representation
    # For our models in the paper, mv=True (will be set to true later in load_emb_in())
    self.mv = False # multi-vector

    if multi:
      self.num_mixtures = 2
      self.emb_multi = np.zeros((self.emb.shape[0], 2, self.emb.shape[1]))
      self.emb_multi_out = np.zeros((self.emb_out.shape[0], 2, self.emb_out.shape[1]))

      self.emb_multi[:,0] = self.emb
      self.emb_multi_out[:,0] = self.emb_out

      if verbose: print "Loading emb2_out"
      self.emb_multi_out[:,1] = self.load_emb_out(out_fname + "2")
      if verbose: print "Loading emb2"
      emb_in2 = self.load_emb_in(in_fname + "2") # this will set mv to true or false, based on the .in2 file
      if self.mv:
        self.emb_multi[:self.nwords, 1] = emb_in2 # the rest of the indices are zero
      else:
        self.emb_multi[:, 1] = emb_in2
      # note: here, multi-vec version has only [num_words] not
      # [num_words + num_buckets]

      # reference - no redundant copies
      self.emb = self.emb_multi[:,0]
      self.emb2 = self.emb_multi[:,1]
      self.emb_out = self.emb_multi_out[:,0]
      self.emb2_out = self.emb_multi_out[:,1]

    # caching
    if self.cache and self.bucket != 0:
      print "Cache subword"
      self.subword_emb = self.cache_subword_rep(basename=basename)
      if multi:
        self.subword_emb_multi = np.zeros((self.subword_emb.shape[0], 2, self.subword_emb.shape[1]))
        self.subword_emb_multi[:, 0] = self.subword_emb
        print "Cache subword2"
        if self.mv:
          # copy from emb_multi directly to subword_emb
          print("No Caching of Multi Rep - subword_emb_multi[:,1] uses dictionary level directly")
          self.subword_emb_multi[:, 1] = self.emb_multi[:self.subword_emb_multi.shape[0], 1]
        else:
          self.subword_emb_multi[:, 1] = self.cache_subword_rep(basename=basename,
            emb = self.emb2,
            suffix=".subword2.npy")

        # reference - no redundant copies
        self.subword_emb = self.subword_emb_multi[:, 0]
        self.subword_emb2 = self.subword_emb_multi[:, 1]
    if savevec:
      fout_vec_name = basename + ".pyvec"
      if False and os.path.isfile(fout_vec_name):
        print "File {} already exists. Not saving".format(fout_vec_name)
        return
      else:
        print "Saving vector file ", fout_vec_name
      fout_vec = open(fout_vec_name, 'w')
      if multi:
        fout_vec1 = open(fout_vec_name + "1", 'w')
        fout_vec2 = open(fout_vec_name + "2", 'w')
        if not os.path.isfile(fout_vec_name):
          fout_vec.write("{} {}\n".format(len(self.id2word), 2*self.emb.shape[1]))
        fout_vec1.write("{} {}\n".format(len(self.id2word), self.emb.shape[1]))
        fout_vec2.write("{} {}\n".format(len(self.id2word), self.emb.shape[1]))

        for _ii, word in enumerate(self.id2word):
          part1 = " ".join(map(str, self.emb[_ii]))
          part2 = " ".join(map(str, self.emb2[_ii]))
          line1 = word + " " + part1 + "\n"
          line2 = word + " " + part2 + "\n"
          if not os.path.isfile(fout_vec_name):
            line = word + " " + part1 + " " + part2 + "\n"
            fout_vec.write(line)
          fout_vec1.write(line1)
          fout_vec2.write(line2)

      else:
        fout_vec.write("{} {}\n".format(len(self.id2word), self.emb.shape[1]))
        for _ii, word in enumerate(self.id2word):
          line = word + " " + " ".join(map(str, self.emb[_ii])) + "\n"
          fout_vec.write(line)

  def load_dict(self, fname):
    self.id2word = []
    with open(fname, 'r') as f:
      for line in f:
        self.id2word.append(line.strip())
    self.word2id = {}
    for i, word in enumerate(self.id2word):
      self.word2id[word] = i
    self.nwords = len(self.id2word)

  def cache_subword_rep(self, basename=None, emb=None, verbose=False, suffix=".subword.npy"):
    if emb is None:
      emb = self.emb
    if basename is not None and os.path.isfile(basename + suffix):
      if verbose: print 'Loading Cached Subword Embeddings'
      subword_emb = np.load(basename + suffix)
      return subword_emb
    if verbose: print 'Caching Subword Embeddings'
    subword_emb = np.zeros((self.nwords, self.D))
    for i, word in enumerate(self.id2word):
      ngram_idxs, _ = self.getNgrams(word)
      if not self.dein:
          ngram_idxs = ngram_idxs[1:]
      if verbose: print ngram_idxs
      for idx in ngram_idxs:
        if verbose >=1: print 'idxs =', idx
        if verbose >=2: print emb[idx]
        subword_emb[i] += emb[idx]
      subword_emb[i] /= len(ngram_idxs)*1.0
    self.cache=True
    if verbose: print 'Saving {} for future read'.format(suffix)
    np.save(basename + suffix, subword_emb)
    return subword_emb

  ############################################################
  def load_emb_out(self, fname, verbose=False):
    if os.path.isfile(fname + '.npy'):
      emb_out = np.load(fname + '.npy')
    else:
      df = pd.read_csv(fname, header=None, delim_whitespace=True,)
      emb_out = df.values
      if verbose: print 'Saving emb_out in .npy for future read'
      np.save(fname + '.npy', emb_out)
    assert emb_out.shape[0] == self.nwords, "emb_out shape = {} and nwords = {}".format(
      emb_out.shape[0], self.nwords)
    self.D = emb_out.shape[1]
    return emb_out

  def load_emb_in(self, fname, verbose=False):
    if os.path.isfile(fname + '.npy'):
      emb = np.load(fname + '.npy')
      if emb.shape[0] == self.nwords:
        self.mv = True
      else:
        self.mv = False
    else:
      df = pd.read_csv(fname, header=None, delim_whitespace=True,)
      emb = df.values
      # checking if this case is mv or not
      if emb.shape[0] == self.nwords:
        print("Setting MV mode = True")
        self.mv = True
      elif emb.shape[0] == self.nwords + self.bucket:
        self.mv = False
      else:
        assert False, "Unexpected error"

      if verbose: print 'Saving emb_in in .npy for future read'
      np.save(fname + '.npy', emb)
    if self.maxn != 0:
      if self.mv:
        assert emb.shape[0] == self.nwords, \
          "[MV mode] shape of loaded emb_in {}/ nwords {} / bucket {} / expected nrows {}".format(emb.shape, 
            self.nwords, 
            self.bucket, 
            self.nwords)
      else:
        assert emb.shape[0] == self.nwords + self.bucket, \
        "shape of loaded emb_in {}/ nwords {} / bucket {} / expected nrows {}".format(emb.shape, 
          self.nwords, 
          self.bucket, 
          self.nwords + self.bucket)

    else:
      assert emb.shape[0] == self.nwords, "For model with maxn=0, we expect the number of rows {} to be the number of words {}".format(
        emb.shape[0], self.nwords)
    assert emb.shape[1] == self.D
    return emb

  ############################################################

  # We also output the original word as part of the "ngrams"
  def getNgrams(self, word):
    ngram_idxs = []
    ngrams = []
    if word in self.word2id:
      ngrams.append(BOW + word + EOW) # do we really need to add the BOW and EOW here? not really
      ngram_idxs.append(self.word2id[word])
    _ngram_hashes, _ngrams = computeNgrams(BOW + word + EOW, minn=self.minn, maxn=self.maxn, bucket=self.bucket)
    ngrams += _ngrams
    ngram_idxs += [self.nwords + hh for hh in _ngram_hashes]
    return ngram_idxs, ngrams
  
  def subword_rep(self, words, emb=None, subword_emb=None, verbose=0):
    if emb is None:
      emb = self.emb
    if subword_emb is None:
      subword_emb = self.subword_emb
    """
    Given a string, obtain the subword representation
    """
    if type(words) is str:
      assert len(words) >= 1
      if words not in self.word2id or not self.cache:
        ngram_idxs, _ = self.getNgrams(words)
        if not self.dein:
          ngram_idxs = ngram_idxs[1:]
        if verbose: print ngram_idxs
        vec = np.zeros((self.D))
        for idx in ngram_idxs:
          if verbose >=1: print 'idxs =', idx
          if verbose >=2: print emb[idx]
          vec += emb[idx]
        vec /= len(ngram_idxs)*1.0
        return vec, 1
      else:
        return subword_emb[self.word2id[words]], 1
    elif type(words) is list:
      vecs = np.zeros((len(words), self.D))
      for i, word in enumerate(words):
        if word not in self.word2id or not self.cache:
          ngram_idxs, _ = self.getNgrams(word)
          # Do not include the word itself (first index) in the average
          if not self.dein:
            ngram_idxs = ngram_idxs[1:]
          for idx in ngram_idxs:
            vecs[i] += emb[idx]
          vecs[i] /= len(ngram_idxs)*1.0
        else:
          vecs[i] = subword_emb[self.word2id[word]]
      return vecs, np.array([1]*len(words), dtype=bool)

  def subword_rep_multi(self, words, verbose=0):
    vecs, _ = self.subword_rep(words)
    vecs2, _ = self.subword_rep(words, emb=self.emb2, subword_emb=self.subword_emb2)
    return np.stack([vecs, vecs2], axis=1), np.array([1]*vecs.shape[0], dtype=bool)

  # only supporting one word lookup
  # If the ngram norm is less than thres, do not add it
  def subword_rep_thres(self, word, verbose=0, thres=0.05):
    if type(word) is str:
      assert len(word) >= 1
      ngram_idxs, _ = self.getNgrams(word)
      # Do not include the word itself (first index) in the average
      if not self.dein:
          ngram_idxs = ngram_idxs[1:]
      vec = np.zeros((self.D))
      counter = 0
      if verbose:
        print "Treshold = ", thres
      for idx in ngram_idxs:
        if np.linalg.norm(self.emb[idx]) > thres:
          vec += self.emb[idx]
          counter += 1
      vec /= (0.000001 + counter*1.0)
      return vec, 1
    elif type(word) is list:
      vecs = np.zeros((len(word), self.D))
      for i in xrange(len(word)):
        vecs[i], _ = self.subword_rep_thres(word[i], thres=thres, verbose=verbose)
      return vecs, np.array([True]*len(word))

  def subword_rep_cos_thres(self, word, verbose=0, thres=0.0):
    if type(word) is str:
      assert len(word) >= 1
      ngram_idxs, _ = self.getNgrams(word)
      # Do not include the word itself (first index) in the average
      if not self.dein:
        ngram_idxs = ngram_idxs[1:]
      vec_temp = np.zeros((self.D))
      for idx in ngram_idxs:
        vec_temp += self.emb[idx]
      vec = np.zeros((self.D))
      counter = 0
      for idx in ngram_idxs:
        cos_sim = 1. - scipy.spatial.distance.cosine(self.emb[idx], vec_temp)
        if cos_sim > thres:
          vec += self.emb[idx]
          counter += 1
      vec /= (0.000001 + counter*1.0)
      if verbose:
        print "Total size {} Num included {}".format(len(ngram_idxs), counter)
      return vec, 1
    elif type(word) is list:
      vecs = np.zeros((len(word), self.D))
      for i in xrange(len(word)):
        vecs[i], _ = self.subword_rep_cos_thres(word[i], thres=thres, verbose=verbose)
      return vecs, np.array([True]*len(word))

  def combined_rep(self, word, verbose=0):
    vec, indic = self.dict_rep(word)
    vec_, _ = self.subword_rep(word)
    vec += vec_
    return vec, indic

  def cross_rep(self, word, verbose=0):
    vec, indic = self.dict_rep_out(word)
    vec_, _ = self.subword_rep(word)
    vec += vec_
    return vec, indic

  def dict_rep(self, words):
    if type(words) is str:
      # if not in the dict, return zero vectors
      vec = np.zeros((self.D))
      if words in self.word2id:
        idx = self.word2id[words]
        vec = self.emb[idx]
        return vec, 1
      else:
        return vec, 0
    elif type(words) is list:
      # return a list of boolean for which vectors are in the dict
      nn = len(words)
      vecs = np.zeros((nn, self.D))
      inds = np.zeros((nn), dtype=bool)
      for i, word in enumerate(words):
        if word in self.word2id:
          idx = self.word2id[word]
          vecs[i] = self.emb[idx]
          inds[i] = True
      return vecs, inds

  def dict_rep2(self, words):
    if type(words) is str:
      # if not in the dict, return zero vectors
      vec = np.zeros((self.D))
      if words in self.word2id:
        idx = self.word2id[words]
        vec = self.emb2[idx]
        return vec, 1
      else:
        return vec, 0
    elif type(words) is list:
      # return a list of boolean for which vectors are in the dict
      nn = len(words)
      vecs = np.zeros((nn, self.D))
      inds = np.zeros((nn), dtype=bool)
      for i, word in enumerate(words):
        if word in self.word2id:
          idx = self.word2id[word]
          vecs[i] = self.emb2[idx]
          inds[i] = True
      return vecs, inds

  def dict_rep_multi(self, words):
    vecs, _ = self.dict_rep(words)
    vecs2, _ = self.dict_rep2(words)
    return np.stack([vecs, vecs2], axis=1), np.array([1]*vecs.shape[0], dtype=bool)  

  def dict_rep_out(self, words, emb_out=None):
    if emb_out is None:
      emb_out = self.emb_out
    if type(words) is str:
      # if not in the dict, return zero vectors
      vec = np.zeros((self.D))
      if words in self.word2id:
        idx = self.word2id[words]
        vec = emb_out[idx]
        return vec, 1
      else:
        return vec, 0
    elif type(words) is list:
      # return a list of boolean for which vectors are in the dict
      nn = len(words)
      vecs = np.zeros((nn, self.D))
      inds = np.zeros((nn), dtype=bool)
      for i, word in enumerate(words):
        if word in self.word2id:
          idx = self.word2id[word]
          vecs[i] = emb_out[idx]
          inds[i] = True
      return vecs, inds

  def dict_rep_out2(self, words):
    return self.dict_rep_out(words, emb_out=self.emb2_out)

  def dict_rep_out_multi(self, words):
    vecs, _ = self.dict_rep_out(words)
    vecs2, _ = self.dict_rep_out2(words)
    return np.stack([vecs, vecs2], axis=1), np.array([1]*vecs.shape[0], dtype=bool)

  def find_nn(self, word, emb_func, limit_vocab_size=30000, num_neighbors=1, verbose=0, show_scores=False):
    # word is a string
    # emb_func is a function that calculates the embedding
    # code you want to evaluate
    start_time = timeit.default_timer()
    if verbose: "word = ", word
    if type(word) is str:
      vec, _ = emb_func(word)
    elif type(word) is np.ndarray:
      assert len(word.shape) == 1
      vec = word
    vec /= (0.000001 + np.linalg.norm(vec))
    search_dict_size = min(len(self.word2id), limit_vocab_size)
    # code you want to evaluate
    elapsed = timeit.default_timer() - start_time
    if verbose: print 'Time before allocation', elapsed
    start_time = timeit.default_timer()
    emb = np.zeros((search_dict_size, self.D))
    elapsed = timeit.default_timer() - start_time
    if verbose: print 'Time for allocation', elapsed
    start_time = timeit.default_timer()
    for i in range(search_dict_size):  
      ww = self.id2word[i]
      emb[i], _ = emb_func(ww)
    elapsed = timeit.default_timer() - start_time
    if verbose: print 'Time for emb calculation', elapsed
    start_time = timeit.default_timer()
    if verbose: print 'Done calculating emb'
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    elapsed = timeit.default_timer() - start_time
    if verbose: print 'Time for normalization', elapsed
    start_time = timeit.default_timer()
    if verbose: print 'Done normalizing'
    cosine_sims = np.dot(emb, vec)
    elapsed = timeit.default_timer() - start_time
    if verbose: print 'Time for dot', elapsed
    start_time = timeit.default_timer()
    if num_neighbors == 1:
      argmax = np.argmax(cosine_sims)
      nn_word = self.id2word[argmax]
      elapsed = timeit.default_timer() - start_time
      if verbose: print 'Time for find max', elapsed
      return nn_word
    else:
      argmin = np.argsort(cosine_sims)
      argmax = argmin[::-1]
      nn_words = np.array(self.id2word)[(argmax[:num_neighbors])]
      if verbose: print 'Done sorting'
      if not show_scores:
        return nn_words
      else:
        scores = cosine_sims[argmax[:num_neighbors]]
        return zip(nn_words, scores)

  def idxs2words(self, idxs):
    # convert a list of strings to a list of words
    words = ["{}:{}".format(self.id2word[idx/self.num_mixtures], idx%self.num_mixtures) for idx in idxs]
    return words

  def show_nearest_neighbors(self, idx_or_word, emb_multi=None, cl=0, num_nns=100, plot=False, verbose=False):
    if emb_multi is None:
      emb_multi = self.subword_emb_multi
    emb_multi = np.reshape(emb_multi, (-1,emb_multi.shape[-1]))

    # Note: emb_multi is a flatten matrix for multi-prototype
    assert isinstance(idx_or_word, int) or idx_or_word in self.word2id, 'Provide index or word in vocabulary'
    idx = idx_or_word
    if idx_or_word in self.word2id:
        idx = self.word2id[idx_or_word]
    dist = np.dot(emb_multi, emb_multi[idx*self.num_mixtures + cl])/(np.linalg.norm(emb_multi, axis=1)*np.linalg.norm(emb_multi[idx*self.num_mixtures + cl]))
    highsim_idxs = dist.argsort()[::-1]
    # select top num_nns (linear) indices with the highest cosine similarity
    highsim_idxs = highsim_idxs[:num_nns]
    dist_val = dist[highsim_idxs]
    words = self.idxs2words(highsim_idxs)
    
    print 'Top highest similarity of {} cl {}'.format(self.id2word[idx], cl)
    print words[:num_nns]
    if verbose: print dist_val[:num_nns]

  def show_nearest_neighbors_single(self, idx_or_word, cl=0, num_nns=100, plot=False, verbose=False):
    # Note: emb_multi is a flatten matrix for multi-prototype
    assert isinstance(idx_or_word, int) or idx_or_word in self.word2id, 'Provide index or word in vocabulary'
    idx = idx_or_word
    if idx_or_word in self.word2id:
        idx = self.word2id[idx_or_word]
    emb = self.subword_emb
    dist = np.dot(emb, emb[idx])/(np.linalg.norm(emb, axis=1)*np.linalg.norm(emb[idx]))
    highsim_idxs = dist.argsort()[::-1]
    # select top num_nns (linear) indices with the highest cosine similarity
    highsim_idxs = highsim_idxs[:num_nns]
    dist_val = dist[highsim_idxs]
    print highsim_idxs
    words = ["{}".format(self.id2word[idx]) for idx in highsim_idxs]
    
    print 'Top highest similarity of {} cl {}'.format(self.id2word[idx], cl)
    print words[:num_nns]
    if verbose: print dist_val[:num_nns]
###################################################################

# read a file and put all the words in a list
def file_to_list(fname):
  assert os.path.isfile(fname)
  f = open(fname, 'r')
  words = []
  for line in f:
    words.append(line.strip())
  return words

# write the embeddings into the vec file format
def output_embs(words, embs):
  assert len(words) == embs.shape[0]
  for word, emb in zip(words, embs):
    print word + " " + " ".join([str(item) for item in emb])

def main():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--subword_model', default='../.ngrams', help='')
  parser.add_argument('--input', default='', help='The query file')
  parser.add_argument('--type', default='subword', choices=['subword', 'dict', 'add'])
  args = parser.parse_args()
  ft = MultiFastText(args.subword_model);
  words = file_to_list(args.input)
  vecs = ft.subword_rep(words)
  output_embs(words, vecs)

if __name__ == "__main__":
  pass