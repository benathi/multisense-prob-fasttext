"""
    Copyright (c) 2018-present. Ben Athiwaratkun
    All rights reserved.
    
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree.
"""
import numpy as np
import pandas as pd
import scipy
import os
#from ggplot import *
import re
from sklearn.metrics import f1_score
from tempfile import NamedTemporaryFile
from gensim.models import KeyedVectors as Word2Vec
from gensim.models.keyedvectors import Vocab
import glob
import multift

## 1. Evaluation Data Loading
def load_SimLex999():
    _fpath = os.path.join('data/word_similarity_datasets/', 'SimLex-999.txt')
    df = pd.read_csv(_fpath, delimiter='\t')
    word1 = df['word1'].tolist()
    word2 = df['word2'].tolist()
    score = df['SimLex999'].tolist()
    assert len(word1) == len(word2) and len(word1) == len(score)
    return word1, word2, score

def load_data_format1(filename='EN-MC-30.txt', delim='\t', verbose=False):
    if verbose: print 'Loading file', filename
    fpath = os.path.join('data/word_similarity_datasets/', filename)
    df = pd.read_csv(fpath, delimiter=delim, header=None)
    word1 = df[0].tolist()
    word2 = df[1].tolist()
    score = df[2].tolist()
    assert len(word1) == len(word2) and len(word1) == len(score)
    return word1, word2, score

def load_data_format3(filename='EN-MC-30.txt', delim='\t', verbose=False, remove_accent=True):
    if verbose: print 'Loading file', filename
    fpath = os.path.join('data/word_similarity_datasets/', filename)
    df = pd.read_csv(fpath, delimiter=delim, header=None)
    word1 = df[0].tolist()
    word2 = df[1].tolist()
    score = df[2].tolist()
    assert len(word1) == len(word2) and len(word1) == len(score)
    if remove_accent:
      word1_ = [filter_accent(_w) for _w in word1]
      word2_ = [filter_accent(_w) for _w in word2]
      return word1_, word2_, score
    else:
      return word1, word2, score

def load_MC():
    return load_data_format1(filename='EN-MC-30.txt')

def load_MEN():
    return load_data_format1(filename='EN-MEN-TR-3k.txt', delim=' ')

def load_Mturk287():
    return load_data_format1(filename='EN-MTurk-287.txt')

def load_Mturk771():
    return load_data_format1(filename='EN-MTurk-771.txt', delim=' ')

def load_RG():
    return load_data_format1(filename='EN-RG-65.txt')

def load_RW_Stanford():
    return load_data_format1(filename='EN-RW-STANFORD.txt')

def load_WS_all():
    return load_data_format1(filename='EN-WS-353-ALL.txt')

def load_WS_rel():
    return load_data_format1(filename='EN-WS-353-REL.txt')

def load_WS_sim():
    return load_data_format1(filename='EN-WS-353-SIM.txt')

def load_YP():
    return load_data_format1(filename='EN-YP-130.txt', delim=' ')


def load_SCWS():
  ll = process_huang()
  words1, _, words2, _, scores = zip(*ll)
  return list(words1), list(words2), list(scores)

def filter_accent(word):
  return ''.join([i if ord(i) < 128 else '' for i in word])

def load_data_format2(filename='foreign/de/gur350.txt', delim=':', verbose=False, header_idx=0,
  remove_accent=True):
  if verbose: print 'Loading file', filename
  fpath = os.path.join('data/word_similarity_datasets/', filename)
  df = pd.read_csv(fpath, delimiter=delim, header=header_idx)
  word1 = df['#WORD1'].tolist()
  word2 = df['WORD2'].tolist()
  score = df['Value'].tolist()
  assert len(word1) == len(word2) and len(word1) == len(score)
  if remove_accent:
    word1_ = [filter_accent(_w) for _w in word1]
    word2_ = [filter_accent(_w) for _w in word2]
    return word1_, word2_, score
  else:
    return word1, word2, score

### For other languages
def load_de_gur350():
  return load_data_format2(filename='foreign/de/gur350.txt')

def load_de_gur65():
  return load_data_format2(filename='foreign/de/gur65.txt')

def load_de_zg222():
  return load_data_format2(filename='foreign/de/zg222.txt')

def load_fr_ws353():
  return load_data_format3(filename='foreign/fr/FR-WS-353.txt', delim=' ')

def load_data_format4(filename='wordsim_foreign_clean/monolingual_lang/it', delim='\t', verbose=False, remove_accent=True):
    if verbose: print 'Loading file', filename
    fpath = os.path.join('data/word_similarity_datasets/', filename)
    df = pd.read_csv(fpath, delimiter=delim, header=0)
    word1 = df["Word1"].tolist()
    word2 = df["Word2"].tolist()
    score = df["Average Score"].tolist()
    assert len(word1) == len(word2) and len(word1) == len(score)
    if remove_accent:
      word1_ = [filter_accent(_w) for _w in word1]
      word2_ = [filter_accent(_w) for _w in word2]
      return word1_, word2_, score
    else:
      return word1, word2, score

# italian
def load_it_ws353():
  return load_data_format4(filename='wordsim_foreign_clean/monolingual_lang/it/MWS353_Italian.txt', delim=',')

def load_it_sl999():
  return load_data_format4(filename='wordsim_foreign_clean/monolingual_lang/it/MSimLex999_Italian.txt', delim=',')


def process_huang(filename='data/word_similarity_datasets/scws.txt',
                context_window=5,
                verbose=False):
  dirname = "/efs/users/benathi/research/fastText"
  filepath = os.path.join(dirname, filename)
  f = open(filepath, 'r')
  result_list = []
  for line_num, line in enumerate(f):
    ob = re.search(r'(.*)<b>(.*)</b>(.*)<b>(.*)</b>(.*?)\t(.+)', line)
    pre1 = ob.group(1).split()
    word1 = ob.group(2).strip()
    middle = ob.group(3).split()
    word2 = ob.group(4).strip()
    post2 = ob.group(5).split()
    scores = ob.group(6).split()
        
    pre1 = pre1[-context_window:]
    post1 = middle[:context_window]
    pre2 = middle[-context_window:]
    post2 = post2[:context_window]
        
    scores = [float(score) for score in scores]
    ave_score = np.mean(np.array(scores))
        
    if verbose:
      print line
      print '---------'
      print 'word {} has context'.format(word1)
      print pre1
      print post1
      print '.........'
      print 'word {} has context'.format(word2)
      print pre2
      print post2
      print 'scores = ', scores
      print 'average score = ', ave_score
    result = (word1, pre1+post1, word2, pre2+post2, ave_score)
    result_list.append(result)
  # each tuple: (word1, context_word1, word2, context_word2, ave_score)
  return result_list


def cosine_sim(v1, v2):
    return 1. - scipy.spatial.distance.cosine(v1 + 0.000001, v2 + 0.000001)

def calculate_correlation(data_loader, vec_gen, verbose=True, lower=False):
    #### data_loader is a function that returns 2 lists of words and the scores
    #### metric is a function that takes w1, w2 and calculate the score
    word1, word2, targets = data_loader()
    if lower:
      word1 = [word.lower() for word in word1]
      word2 = [word.lower() for word in word2]

    if verbose:
      print "*************************************"
      print type(word1)
      print len(word1)
      print word1[0]
    word1_idxs, indic1 = vec_gen(word1)
    word2_idxs, indic2 = vec_gen(word2)
    num_oov = sum(indic1) + sum(indic2)
    num_total = len(word1)*2
    if verbose: print "# of words: total {} OOV {}".format(num_total, num_total - num_oov)

    scores = np.zeros((len(word1_idxs)))
    for _i, [w1, w2] in enumerate(zip(word1_idxs, word2_idxs)):
      if len(w1.shape) == 1:
        scores[_i] = cosine_sim(w1, w2)
      elif len(w1.shape) == 2:
        if verbose: print "Using Maximum Cosine Similarity"
        # Using the maximum cosine similarity
        scores[_i] = max([cosine_sim(i,j) for i in w1 for j in w2])
         
    spr = scipy.stats.spearmanr(scores, targets)
    if verbose: print 'Spearman correlation is {} with pvalue {}'.format(spr.correlation, spr.pvalue)
    pear = scipy.stats.pearsonr(scores, targets)
    if verbose: print 'Pearson correlation', pear
    spr_correlation = spr.correlation
    pear_correlation = pear[0]
    if np.any(np.isnan(scores)):
        spr_correlation = np.NAN
        pear_correlation = np.NAN
    return scores, spr_correlation, pear_correlation, num_total, num_oov

### global variables ###
wordsim_dataset_funcs = [load_SimLex999, load_WS_all, load_WS_sim, load_WS_rel, 
                load_MEN, load_MC, load_RG, load_YP,
                load_Mturk287, load_Mturk771,
                load_RW_Stanford, load_SCWS,
                load_de_gur350, load_de_gur65, load_de_zg222, load_fr_ws353, load_it_ws353, load_it_sl999]

wordsim_dataset_names = ['SL', 'WS', 'WS-S', 'WS-R', 'MEN',
                             'MC', 'RG', 'YP', 'MT-287', 'MT-771', 'RW', 'SCWS',
                             'DE-GUR350', 'DE-GUR65', 'DE-ZG222', 'FR-WS-353', 'IT-WS-353', 'IT-SL-999']
wordsim_dict = {}
for name, func in zip(wordsim_dataset_names, wordsim_dataset_funcs):
  wordsim_dict[name] = func

def get_ws_loader(ds):
  assert ds in wordsim_dict
  return wordsim_dict[ds]

# We change it to lower case by default
def wordsim_eval(model_names,
  wordsim_datasets=['SL', 'WS', 'WS-S', 'WS-R', 'MEN', 'MC', 'RG', 'YP', 'MT-287', 'MT-771', 'RW'],
  lower=True,
  verbose=0):
  spearman_corrs = pd.DataFrame()
  assert set(wordsim_datasets) < set(wordsim_dataset_names) # test that it's a subset
  spearman_corrs['Dataset'] = wordsim_datasets
  dir_path = os.path.dirname(os.path.realpath(__file__))
  for i, model in enumerate(model_names):
    model_abbrev, vector_gen = model[:2]
    results = []
    for wordsim_ds in wordsim_datasets:
      ws_loader = get_ws_loader(wordsim_ds)
      _, sp, pe, num_total, num_oov = calculate_correlation(ws_loader, vector_gen, lower=lower, verbose=verbose)
      results.append(sp*100)
    colname = '{}'.format(model_abbrev)
    spearman_corrs[colname] = results
  return spearman_corrs

def flatten_list(ll):
    return  [item for sublist in ll for item in sublist]

def prep_embeddings_fast(ft, emb_func, limit_vocab_size=30000):
  w2v = Word2Vec()
  emb = None
  if type(emb_func) is np.ndarray:
    size = min(limit_vocab_size, emb_func.shape[0])
    emb =  emb_func[:size]
  else:
    emb = np.zeros((limit_vocab_size, ft.D))
    for i in range(limit_vocab_size):
      emb[i], _ = emb_func(ft.id2word[i])
  size = emb.shape[0]
  w2v.index2word = ft.id2word[:size]
  w2v.vector_size = ft.D
  w2v.syn0 = emb
  dvocab = {}
  for word_id, word in enumerate(w2v.index2word):
    dvocab[word] = Vocab(index=word_id, count=ft.nwords - word_id)
  w2v.vocab = dvocab
  return w2v

def print_accuracy(model, questions_file, verbose=True):
  acc = model.accuracy(questions_file)
  sem_correct = sum((len(acc[i]['correct']) for i in range(5)))
  sem_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5))
  sem_acc = 100*float(sem_correct)/sem_total
  if verbose: print('Semantic Analogy\t: {:.2f} ({:d}/{:d}) (OOV: ?)'.format(sem_acc, sem_correct, sem_total))
    
  syn_correct = sum((len(acc[i]['correct']) for i in range(5, len(acc)-1)))
  syn_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5,len(acc)-1))
  syn_acc = 100*float(syn_correct)/syn_total
  if verbose: print('Syntactic Analogy\t: {:.2f} ({:d}/{:d}) (OOV: ?)'.format(syn_acc, syn_correct, syn_total))
  return (sem_acc, syn_acc)

def analogy_eval_gensim_single(words, emb_func, ft=None, D=100, limit_vocab_size=30000, q_fname='/home/ubuntu/research/groupSparsityFastText/data/questions-words.txt', debug=False, verbose=0):
  if verbose: print 'Using prep_embeddings_fast'
  w2v_model = prep_embeddings_fast(ft, emb_func, limit_vocab_size)
  sem_acc, syn_acc = print_accuracy(w2v_model, q_fname, verbose)
  return sem_acc, syn_acc

def analogy_eval(models, limit_vocab_size=30000, verbose=0):
  # (abbev_model_name, ft, emb_func/emb)
  df = pd.DataFrame()
  df['Dataset'] = ['SemAna', 'SynAna']
  for model in models:
    name, emb_func, ft = model[:3]
    if len(model) > 3:
      emb_func = model[3]
    sem_acc, syn_acc = analogy_eval_gensim_single(ft.id2word, emb_func, ft=ft, D=ft.D, limit_vocab_size=min(limit_vocab_size, ft.nwords), verbose=verbose)
    df[name] = [sem_acc, syn_acc]
  return df

def norm_eval(models):
  # print out the average norm of the embeddings
  df = pd.DataFrame()
  df['Dataset'] = ['Norm-Mean', 'Norm-Std']
  for model in models:
    # expect (name, emb, ft*[if emb is rep])
    name = model[0]
    emb = model[1]
    if len(model) > 2:
      ft = model[2]
    if not(type(emb) == np.ndarray):
      assert ft is not None
      emb_ = np.zeros((ft.nwords, ft.D))
      for i in range(ft.nwords):
        emb_[i], _ = emb(ft.id2word[i])
    else:
      emb_ = emb
    norms = np.linalg.norm(emb_, axis=1, keepdims=False)
    mean = np.mean(norms)
    std = np.std(norms)
    df[name] = [mean, std]
  return df

def get_list_basenames(pattern, add_dot=False):
  # adding dot - getting around the issue of '.' in model name
  list_files = glob.glob(pattern)
  list_files_2 = [path.split('.')[0] for path in list_files]
  if not add_dot:
    print "Not Adding Dot -----"
    return list(set(list_files_2))
  else:
    print("Adding dot")
    ll = list(set(list_files_2))
    ll = [item +"." for item in ll]
    return ll

# Get the fts objects by a list of specified basenames or a matching pattern (glob pattern)
def get_fts(list_basenames_or_pattern, verbose=0, multi=False, ft=True):
  if type(list_basenames_or_pattern) is str:
    list_basenames = get_list_basenames(list_basenames_or_pattern)
  else:
    list_basenames = list_basenames_or_pattern
  fts = []
  for l in list_basenames:
    # set maxn=0 if the we do not use the ft structure (ft=0)
    maxn=6
    if not ft:
      maxn=0
    ft = multift.MultiFastText(basename=l, verbose=verbose, multi=multi, maxn=maxn)
    fts.append(ft)
  return fts

# performs quantitative evaluation in a batch
# (1) word similirity
# (2) semantic and syntactic analogy (look at gensim)
# (3) downstream tasks - pending
def quantitative_eval(model_names, verbose=0):
  # model_names is list of (name, emb_func, ft, emb*)
  names, emb_funcs, _, _ = zip(*model_names)
  model_names_2 = zip(names, emb_funcs)
  df0 = norm_eval(model_names)
  if verbose: print df0
  df1 = wordsim_eval(model_names_2)
  if verbose: print df1
  df2 = analogy_eval(model_names)
  if verbose: print df2
  df0 = df0.append(df1)
  df0 = df10.append(df2)
  return df0

def eval_ft(ft, name, verbose=0, which=['wordsim'], wordrep=['sub'],
  wordsim_datasets=['SL', 'WS', 'WS-S', 'WS-R', 'MEN', 'MC', 'RG', 'YP', 'MT-287', 'MT-771', 'RW'],
  lower=True):
  # wordrep=['sub', 'dout', 'din', 'combined']
  print 'wordrep =', wordrep
  model_names = []
  for rep in wordrep:
    _name = name + '-' + rep
    tup = None
    if rep == 'sub':
      tup = (_name, ft.subword_rep, ft, ft.subword_emb)
    elif rep == 'dout':
      tup = (_name, ft.dict_rep_out, ft, ft.emb_out)
    elif rep == 'din':
      tup = (_name, ft.dict_rep, ft, ft.emb[:ft.nwords])
    elif 'sub-thres' in rep:
      thres_val = float(rep[len('sub-thres'):])
      if verbose: print 'Thres val =', thres_val
      tup = (_name, lambda x: ft.subword_rep_thres(x, thres=thres_val), ft, None)
    elif rep == 'combined':
      tup = (_name, ft.combined_rep, ft, ft.subword_emb + ft.emb[:ft.nwords])
    ######
    model_names.append(tup)

  df = pd.DataFrame()
  if 'analogy' in which:
    if verbose: print 'performing Analogy'
    _df = analogy_eval(model_names, verbose=verbose)
    df = df.append(_df)
  if 'wordsim' in which:
    if verbose: print 'performing wordsim'
    _df = wordsim_eval(model_names, verbose=verbose, wordsim_datasets=wordsim_datasets, lower=lower)
    df = df.append(_df)
  if 'norm' in which:
    if verbose: print 'performing norm analysis'
    _df = norm_eval(model_names)
    df = df.append(_df)
  return df

def eval_fts(fts, names, verbose=0, transpose=True, which=['norm', 'wordsim'], wordrep=['sub', 'dout', 'din', 'combined'],
  wordsim_datasets=['WS', 'MEN', 'RW', 'SCWS'], lower=True):
  big_df = None
  for ft, name in zip(fts, names):
    df = eval_ft(ft, name, which=which, verbose=verbose, wordrep=wordrep, lower=lower)
    if verbose: print df
    if big_df is None:
      big_df = df
    else:
      big_df = pd.merge(big_df, df)
  if transpose: 
    return big_df.transpose(), fts
  else: 
    return big_df, fts

def eval_dir(dir_pattern, verbose=0, which=['wordsim'], wordrep=['sub', 'dout', 'din', 'combined'],
  add_dot=False, lower=True, multi=False,
  wordsim_datasets=['SL', 'WS', 'WS-S', 'WS-R', 'MEN', 'MC', 'RG', 'YP', 'MT-287', 'MT-771', 'RW', 'SCWS'], ft=True):
  if type(dir_pattern) is str:
    list_names = get_list_basenames(dir_pattern, add_dot=add_dot)
  elif type(dir_pattern) is list:
    list_names = dir_pattern
  if verbose:
    print 'The basenames to evaluate:'
    for name in list_names:
      print name
  abbrev_names = [item.split("/")[-1] for  item in list_names]
  dfs = []
  for basename in list_names:
    print "\n\n------------------------------------------"
    print "Evaluating Basename", basename
    print "ft=", ft
    if ft:
      wordrep=['sub', 'dout', 'din', 'combined']
    else:
      wordrep=['dout', 'din']
    ft = get_fts([basename], multi=multi, ft=ft)[0]
    print("wordrep=", wordrep)
    if multi:
      df = eval_multi(ft, lower=lower)
    else:
      df = eval_ft(ft, basename, lower=lower, wordsim_datasets=wordsim_datasets, wordrep=wordrep)
    print "df ="
    print df
    dfs.append((basename, df))
  return dfs

def plot_norm(emb):
  # Given an ft, plot the norm of the embeddings
  norms = np.linalg.norm(emb, axis=1, keepdims=False)
  df = pd.DataFrame({'X':range(emb.shape[0]), 'Y':norms})
  p = (ggplot(aes(x='X', y='Y'), data=df) 
      + geom_point(alpha=0.01)
      + ylim(low=0, high=10)
    )
  print p

def emb_norm_plots(ft, name=None):
  if hasattr(ft, 'basename'):
    print "Basename =", ft.basename
  elif name is not None:
    print "Name = ", name
  print 'Emb'
  plot_norm(ft.emb)
  print 'Emb Out'
  plot_norm(ft.emb_out)
  print 'Subword Emb'
  plot_norm(ft.subword_emb)

def eval_multi(_ft, verbose=False, lower=True, wordsim_datasets=['SL', 'WS', 'WS-S', 'WS-R', 'MEN', 'MC', 'RG', 'YP', 'MT-287', 'MT-771', 'RW']):
  return wordsim_eval([('sub', _ft.subword_rep),
                      ('sub2', lambda word: _ft.subword_rep(word, emb=_ft.emb2, subword_emb=_ft.subword_emb2)),
                      ('sub-maxsim', _ft.subword_rep_multi),
                      #('out', _ft.dict_rep_out),
                      #('out2', _ft.dict_rep_out2),
                      #('out-maxsim', _ft.dict_rep_out_multi)
                     ], verbose=verbose,
                    wordsim_datasets=wordsim_datasets, lower=lower)

def eval_multi_word(_ft, verbose=False, lower=True, wordsim_datasets=['SL', 'WS', 'WS-S', 'WS-R', 'MEN', 'MC', 'RG', 'YP', 'MT-287', 'MT-771', 'RW']):
  return wordsim_eval([ ('in', _ft.dict_rep),
                        ('in2', _ft.dict_rep2),
                        ('in-maxsim', _ft.dict_rep_multi),
                      #('out', _ft.dict_rep_out),
                      #('out2', _ft.dict_rep_out2),
                      #('out-maxsim', _ft.dict_rep_out_multi)
                     ], verbose=verbose,
                    wordsim_datasets=wordsim_datasets, lower=lower)

def test_nn(ft, verbose=False, rep='subword', plot=False, num_nns=100):
  word = 'rock'
  if rep == 'subword':
      ft.show_nearest_neighbors(word, cl=0, emb_multi=ft.subword_emb_multi, verbose=verbose, plot=plot, num_nns=num_nns)
      ft.show_nearest_neighbors(word, cl=1, emb_multi=ft.subword_emb_multi, verbose=verbose, plot=plot, num_nns=num_nns)
  else:
      ft.show_nearest_neighbors(word, cl=0, emb_multi=ft.emb_multi_out, verbose=verbose, plot=plot, num_nns=num_nns)
      ft.show_nearest_neighbors(word, cl=1, emb_multi=ft.emb_multi_out, verbose=verbose, plot=plot, num_nns=num_nns)
  print "----------------------------------"
  word = 'bank'
  if rep == 'subword':
      ft.show_nearest_neighbors(word, cl=0, emb_multi=ft.subword_emb_multi, verbose=verbose, plot=plot, num_nns=num_nns)
      ft.show_nearest_neighbors(word, cl=1, emb_multi=ft.subword_emb_multi, verbose=verbose, plot=plot, num_nns=num_nns)
  else:
      ft.show_nearest_neighbors(word, cl=0, emb_multi=ft.emb_multi_out, verbose=verbose, plot=plot, num_nns=num_nns)
      ft.show_nearest_neighbors(word, cl=1, emb_multi=ft.emb_multi_out, verbose=verbose, plot=plot, num_nns=num_nns)
  print "----------------------------------"
  word = 'star'
  if rep == 'subword':
      ft.show_nearest_neighbors(word, cl=0, emb_multi=ft.subword_emb_multi, verbose=verbose, plot=plot, num_nns=num_nns)
      ft.show_nearest_neighbors(word, cl=1, emb_multi=ft.subword_emb_multi, verbose=verbose, plot=plot, num_nns=num_nns)
  else:
      ft.show_nearest_neighbors(word, cl=0, emb_multi=ft.emb_multi_out, verbose=verbose, plot=plot, num_nns=num_nns)
      ft.show_nearest_neighbors(word, cl=1, emb_multi=ft.emb_multi_out, verbose=verbose, plot=plot, num_nns=num_nns)
  print "----------------------------------"
  word = 'left'
  if rep == 'subword':
      ft.show_nearest_neighbors(word, cl=0, emb_multi=ft.subword_emb_multi, verbose=verbose, plot=plot, num_nns=num_nns)
      ft.show_nearest_neighbors(word, cl=1, emb_multi=ft.subword_emb_multi, verbose=verbose, plot=plot, num_nns=num_nns)
  else:
      ft.show_nearest_neighbors(word, cl=0, emb_multi=ft.emb_multi_out, verbose=verbose, plot=plot, num_nns=num_nns)
      ft.show_nearest_neighbors(word, cl=1, emb_multi=ft.emb_multi_out, verbose=verbose, plot=plot, num_nns=num_nns)


if __name__=="__main__":
  pass