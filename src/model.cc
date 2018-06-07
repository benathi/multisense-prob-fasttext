/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *               2018-present, Ben Athiwaratkun
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <random>
#include <cmath>

namespace fasttext {

Model::Model(std::shared_ptr<Matrix> wi,
             std::shared_ptr<Matrix> wo,
             std::shared_ptr<Matrix> wi2,
             std::shared_ptr<Matrix> wo2,
             // for variance
             std::shared_ptr<Matrix> invar,
             std::shared_ptr<Matrix> invar2,
             std::shared_ptr<Matrix> outvar,
             std::shared_ptr<Matrix> outvar2,
             std::shared_ptr<Args> args,
             int32_t seed)
  : hidden_(args->dim), hidden2_(args->dim), output_(wo->m_),
  grad_(args->dim), grad2_(args->dim), temp_(args->dim), gradvar_(args->dim),gradvar2_(args->dim), rng(seed), quant_(false)
{
  wi_ = wi;
  wo_ = wo;
  wi2_ = wi2;
  wo2_ = wo2;

  invar_ = invar;
  invar2_ = invar2;
  outvar_ = outvar;
  outvar2_ = outvar2;

  args_ = args;
  osz_ = wo->m_;
  hsz_ = args->dim;
  negpos = 0;
  loss_ = 0.0;
  nexamples_ = 1;
  initSigmoid();
  initLog();
}

Model::~Model() {
  delete[] t_sigmoid;
  delete[] t_log;
}

void Model::setQuantizePointer(std::shared_ptr<QMatrix> qwi,
                               std::shared_ptr<QMatrix> qwo, bool qout) {
  qwi_ = qwi;
  qwo_ = qwo;
  if (qout) {
    osz_ = qwo_->getM();
  }
}

real Model::binaryLogistic(int32_t target, bool label, real lr) {
  real score = sigmoid(wo_->dotRow(hidden_, target));
  real alpha = lr * (real(label) - score);
  grad_.addRow(*wo_, target, alpha);
  wo_->addRow(hidden_, target, alpha);
  if (label) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

real Model::elk(int32_t target, bool label, real lr) {
  // not used
  return 0.0;
}

real Model::negativeSampling(int32_t target, real lr) {
  // loss is the negative of similarity here
  grad_.zero();
  real sim1 = 0.0;
  real sim2 = 0.0;
  real scale = lr/(args_->var_scale);
  int32_t negTarget = getNegative(target);

  // We're not using the method ELK

  hidden_.addRow(*wo_, target, -1.); // mu - v_out
  sim1 = - (1./args_->var_scale)*(hidden_.normsq());
  hidden_.addRow(*wo_, target, 1.); // mu
  hidden_.addRow(*wo_, negTarget, -1.); // mu - v_out_neg
  sim2 = - (1./args_->var_scale)*(hidden_.normsq());
  hidden_.addRow(*wo_, negTarget, 1.); // mu

  real loss = args_->margin - sim1 + sim2;
  if (loss > 0.0){
    // This is the only case where we would update the vectors
    grad_.addRow(*wo_, target, scale);
    grad_.addRow(*wo_, negTarget, -scale);
    // Update wo_ itself
    hidden_.addRow(*wo_, target, -1.); // mu - v_out
    // calculate the loss based on the norm
    wo_->addRow(hidden_, target, scale); // 
    hidden_.addRow(*wo_, target, 1.); // mu
    hidden_.addRow(*wo_, negTarget, -1.); // mu - v_out_neg
    wo_->addRow(hidden_, negTarget, -scale);
    hidden_.addRow(*wo_, negTarget, 1.); // mu
  }
  return std::max((real) 0.0, loss);
}

real Model::negativeSamplingSingleExpdot(int32_t target, real lr) {
  // loss is the negative of similarity here
  grad_.zero();
  real sim1 = 0.0;
  real sim2 = 0.0;
  real scale = lr/(args_->var_scale);
  int32_t negTarget = getNegative(target);

  sim1 = wo_->dotRow(hidden_, target);
  sim2 = wo_->dotRow(hidden_, negTarget);

  real loss = args_->margin - sim1 + sim2;
  if (loss > 0.0){
    grad_.addRow(*wo_, target, scale);
    grad_.addRow(*wo_, negTarget, -scale);
    
    // Update wo_ itself
    // calculate the loss based on the norm
    wo_->addRow(hidden_, target, scale);
    wo_->addRow(hidden_, negTarget, -scale);
  }
  return std::max((real) 0.0, loss);
}


// This is for multiple prototype!
real Model::partial_energy(Vector& hidden, Vector& grad, std::shared_ptr<Matrix> wo, int32_t target){
  hidden.addRow(*wo, target, -1.); // mu - v_out
  real sim = - (0.5/args_->var_scale)*(hidden.normsq());
  hidden.addRow(*wo, target, 1.); // mu
  return sim;
}

std::vector<float> Model::energy(int32_t target){
  real sim = 0.0;
  real sim00 = 0.0;
  real sim01 = 0.0;
  real sim10 = 0.0;
  real sim11 = 0.0;

  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      if (i==0 and j ==0){
        sim00 = partial_energy(hidden_, grad_, wo_, target);
      } else if (i==0 and j==1){ 
        sim01 = partial_energy(hidden_, grad_, wo2_, target);
      } else if (i==1 and j==0){
        sim10 = partial_energy(hidden2_, grad2_, wo_, target);
      } else if (i==1 and j==1){
        sim11 = partial_energy(hidden2_, grad2_, wo2_, target);
      }
    }
  }

  std::vector<float> pes;
  pes.push_back(sim00);
  pes.push_back(sim01);
  pes.push_back(sim10);
  pes.push_back(sim11);
  auto it = max_element(std::begin(pes), std::end(pes));
  float max_pe = *it;
  float sum_exp_diff = 0.0;
  for (auto it = pes.begin(); it != pes.end(); ++it){
    sum_exp_diff += std::exp( (*it) - max_pe);
  }
  // We return [max_partial_energy, sum of e(xi_ij - max_pe), the energy]
  std::vector<float> result;
  result.push_back(max_pe);
  result.push_back(sum_exp_diff);
  result.push_back(max_pe + std::log(sum_exp_diff)); // This is the loss
  result.push_back(std::exp(sim00 - max_pe));
  result.push_back(std::exp(sim01 - max_pe));
  result.push_back(std::exp(sim10 - max_pe));
  result.push_back(std::exp(sim11 - max_pe));
  return result;
}

// partial energy expdot
real Model::partial_energy_expdot(Vector& hidden, Vector& grad, std::shared_ptr<Matrix> wo, int32_t target){
  real sim = wo->dotRow(hidden, target);
  return args_->var_scale*sim;
}

// This should be the same as the energy expdot actually
std::vector<float> Model::energy_expdot(int32_t target){
  real sim = 0.0;
  real sim00 = 0.0;
  real sim01 = 0.0;
  real sim10 = 0.0;
  real sim11 = 0.0;

  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      if (i==0 and j ==0){
        sim00 = partial_energy_expdot(hidden_, grad_, wo_, target);
      } else if (i==0 and j==1){ 
        sim01 = partial_energy_expdot(hidden_, grad_, wo2_, target);
      } else if (i==1 and j==0){
        sim10 = partial_energy_expdot(hidden2_, grad2_, wo_, target);
      } else if (i==1 and j==1){
        sim11 = partial_energy_expdot(hidden2_, grad2_, wo2_, target);
      }
    }
  }

  std::vector<float> pes;
  pes.push_back(sim00);
  pes.push_back(sim01);
  pes.push_back(sim10);
  pes.push_back(sim11);
  auto it = max_element(std::begin(pes), std::end(pes));
  float max_pe = *it;
  float sum_exp_diff = 0.0;
  for (auto it = pes.begin(); it != pes.end(); ++it){
    sum_exp_diff += std::exp( (*it) - max_pe);
  }
  // We return [max_partial_energy, sum of e(xi_ij - max_pe), the energy]
  std::vector<float> result;
  result.push_back(max_pe);
  result.push_back(sum_exp_diff);
  result.push_back(max_pe + std::log(sum_exp_diff)); // This is the loss
  result.push_back(std::exp(sim00 - max_pe));
  result.push_back(std::exp(sim01 - max_pe));
  result.push_back(std::exp(sim10 - max_pe));
  result.push_back(std::exp(sim11 - max_pe));
  return result;
}

// partial energy expdot
real Model::partial_energy_vecvar(Vector& hidden, Vector& grad, std::shared_ptr<Matrix> wo, int32_t wordidx, int32_t target, std::shared_ptr<Matrix> varin, std::shared_ptr<Matrix> varout){
  temp_.zero();
  for (int64_t j = 0; j < varin->n_; j++){
    temp_.data_[j] += exp(varin->at(wordidx, j)) + exp(varout->at(target, j));
  }
  hidden_.addRow(*wo, target, -1.);
  real sim = 0.0;
  for (int64_t i = 0; i < temp_.m_; i++) {
    sim += pow(hidden_.data_[i], 2.0)/(1e-8 + temp_.data_[i]);
    sim += log(temp_.data_[i]); // This is the log det part
  }
  sim *= -0.5;
  hidden.addRow(*wo, target, 1.); // mu
  return sim;
}

// This should be the same as the energy expdot actually
std::vector<float> Model::energy_vecvar(int32_t wordidx, int32_t target){
  real sim = 0.0;
  real sim00 = 0.0;
  real sim01 = 0.0;
  real sim10 = 0.0;
  real sim11 = 0.0;

  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      if (i==0 and j ==0){
        sim00 = partial_energy_vecvar(hidden_, grad_, wo_, wordidx, target, invar_, outvar_);
      } else if (i==0 and j==1){
        sim01 = partial_energy_vecvar(hidden_, grad_, wo2_, wordidx, target, invar_, outvar2_);
      } else if (i==1 and j==0){
        sim10 = partial_energy_vecvar(hidden2_, grad2_, wo_, wordidx, target, invar2_, outvar_);
      } else if (i==1 and j==1){
        sim11 = partial_energy_vecvar(hidden2_, grad2_, wo2_, wordidx, target, invar2_, outvar2_);
      }
    }
  }

  std::vector<float> pes;
  pes.push_back(sim00);
  pes.push_back(sim01);
  pes.push_back(sim10);
  pes.push_back(sim11);
  auto it = max_element(std::begin(pes), std::end(pes));
  float max_pe = *it;
  float sum_exp_diff = 0.0;
  for (auto it = pes.begin(); it != pes.end(); ++it){
    sum_exp_diff += std::exp( (*it) - max_pe);
  }
  // We return [max_partial_energy, sum of e(xi_ij - max_pe), the energy]
  std::vector<float> result;
  result.push_back(max_pe);
  result.push_back(sum_exp_diff);
  result.push_back(max_pe + std::log(sum_exp_diff)); // This is the loss
  result.push_back(std::exp(sim00 - max_pe));
  result.push_back(std::exp(sim01 - max_pe));
  result.push_back(std::exp(sim10 - max_pe));
  result.push_back(std::exp(sim11 - max_pe));
  return result;
}



real Model::negativeSamplingMulti(int32_t target, real lr){
  grad_.zero();
  grad2_.zero();
  // 1. we compute sim1 and sim2 and see if we need to update
  std::vector<float> eplus_result = energy(target);
  int32_t negTarget = getNegative(target);
  std::vector<float> eminus_result = energy(negTarget);
  real loss = args_->margin - eplus_result.at(2) + eminus_result.at(2);
  if (loss > 0.0){
    // 2. The goal is to update grad_ and grad2_
    real inv_sum_eplus = lr*(1./eplus_result.at(1))*(-1./args_->var_scale); // plus
    real inv_sum_eminus = lr*(1./eminus_result.at(1))*(-1./args_->var_scale); // minus
    
    std::vector< std::vector<real> > xi_plus;
    std::vector<real> vp1;
    std::vector<real> vp2;
    vp1.push_back(eplus_result.at(3));
    vp1.push_back(eplus_result.at(4));
    vp2.push_back(eplus_result.at(5));
    vp2.push_back(eplus_result.at(6));
    xi_plus.push_back(vp1);
    xi_plus.push_back(vp2);
    
    std::vector< std::vector<real> > xi_minus;
    std::vector<real> vm1;
    std::vector<real> vm2;
    vm1.push_back(eminus_result.at(3));
    vm1.push_back(eminus_result.at(4));
    vm2.push_back(eminus_result.at(5));
    vm2.push_back(eminus_result.at(6));
    xi_minus.push_back(vm1);
    xi_minus.push_back(vm2);

    // (1) Update grad_
    // Do it for context j+
    // j=0
    grad_.addVector(hidden_, xi_plus.at(0).at(0)*inv_sum_eplus);
    grad_.addRow(*wo_, target, -xi_plus.at(0).at(0)*inv_sum_eplus);
    // j=1
    grad_.addVector(hidden_, xi_plus.at(0).at(1)*inv_sum_eplus);
    grad_.addRow(*wo2_, target, -xi_plus.at(0).at(1)*inv_sum_eplus);

    // Do it for context j-
    grad_.addVector(hidden_, -xi_minus.at(0).at(0)*inv_sum_eminus);
    grad_.addRow(*wo_, negTarget, xi_minus.at(0).at(0)*inv_sum_eminus);
    // j=1
    grad_.addVector(hidden_, -xi_minus.at(0).at(1)*inv_sum_eminus);
    grad_.addRow(*wo2_, negTarget, xi_minus.at(0).at(1)*inv_sum_eminus);

    // (2) Update grad2_
    grad2_.addVector(hidden2_, xi_plus.at(1).at(0)*inv_sum_eplus);
    grad2_.addRow(*wo_, target, -xi_plus.at(1).at(0)*inv_sum_eplus);
    // j=1
    grad2_.addVector(hidden2_, xi_plus.at(1).at(1)*inv_sum_eplus);
    grad2_.addRow(*wo2_, target, -xi_plus.at(1).at(1)*inv_sum_eplus);

    // Do it for context j-   
    grad2_.addVector(hidden2_, -xi_minus.at(1).at(0)*inv_sum_eminus);
    grad2_.addRow(*wo_, negTarget, xi_minus.at(1).at(0)*inv_sum_eminus);
    // j=1
    grad2_.addVector(hidden2_, -xi_minus.at(1).at(1)*inv_sum_eminus);
    grad2_.addRow(*wo2_, negTarget, xi_minus.at(1).at(1)*inv_sum_eminus);

    ///////////////////////////////
    // (3) Update wo_[target]     --- this involves eplus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_plus.at(0).at(0));
    temp_.addRow(*wo_, target, xi_plus.at(0).at(0));
    // from i=1
    temp_.addVector(hidden2_, -xi_plus.at(1).at(0));
    temp_.addRow(*wo_, target, xi_plus.at(1).at(0));
    wo_->addRow(temp_, target, inv_sum_eplus);
    
    // (4) Update wo2[target]     --- this involves eplus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_plus.at(0).at(1));
    temp_.addRow(*wo2_, target, xi_plus.at(0).at(1));
    // from i=1
    temp_.addVector(hidden2_, -xi_plus.at(1).at(1));
    temp_.addRow(*wo2_, target, xi_plus.at(1).at(1));
    wo2_->addRow(temp_, target, inv_sum_eplus);

    // (5) Update wo_[negTarget]  --- this involves eminus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_minus.at(0).at(0));
    temp_.addRow(*wo_, negTarget, xi_minus.at(0).at(0));
    // from i=1
    temp_.addVector(hidden2_, -xi_minus.at(1).at(0));
    temp_.addRow(*wo_, negTarget, xi_minus.at(1).at(0));
    wo_->addRow(temp_, negTarget, -inv_sum_eminus);
    
    // (6) Update wo2_[negTarget] --- this involves eminus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_minus.at(0).at(1));
    temp_.addRow(*wo2_, negTarget, xi_minus.at(0).at(1));
    // from i=1
    temp_.addVector(hidden2_, -xi_minus.at(1).at(1));
    temp_.addRow(*wo2_, negTarget, xi_minus.at(1).at(1));
    wo2_->addRow(temp_, negTarget, -inv_sum_eminus);
  }
  return std::max((real) 0.0, loss);
}

real Model::negativeSamplingMultiVec2(int32_t target, real lr){
  grad_.zero();
  grad2_.zero();
  // 1. we compute sim1 and sim2 and see if we need to update
  std::vector<float> eplus_result = energy(target);
  int32_t negTarget = getNegative(target);
  std::vector<float> eminus_result = energy(negTarget);
  real loss = args_->margin - eplus_result.at(2) + eminus_result.at(2);
  if (loss > 0.0){
    // 2. The goal is to update grad_ and grad2_
    real inv_sum_eplus = lr*(1./eplus_result.at(1))*(-1./args_->var_scale); // plus
    real inv_sum_eminus = lr*(1./eminus_result.at(1))*(-1./args_->var_scale); // minus
    
    std::vector< std::vector<real> > xi_plus;
    std::vector<real> vp1;
    std::vector<real> vp2;
    vp1.push_back(eplus_result.at(3));
    vp1.push_back(eplus_result.at(4));
    vp2.push_back(eplus_result.at(5));
    vp2.push_back(eplus_result.at(6));
    xi_plus.push_back(vp1);
    xi_plus.push_back(vp2);
    
    std::vector< std::vector<real> > xi_minus;
    std::vector<real> vm1;
    std::vector<real> vm2;
    vm1.push_back(eminus_result.at(3));
    vm1.push_back(eminus_result.at(4));
    vm2.push_back(eminus_result.at(5));
    vm2.push_back(eminus_result.at(6));
    xi_minus.push_back(vm1);
    xi_minus.push_back(vm2);

    // (1) Update grad_
    // Do it for context j+
    // j=0
    grad_.addVector(hidden_, xi_plus.at(0).at(0)*inv_sum_eplus);
    grad_.addRow(*wo_, target, -xi_plus.at(0).at(0)*inv_sum_eplus);
    // j=1
    grad_.addVector(hidden_, xi_plus.at(0).at(1)*inv_sum_eplus);
    grad_.addRow(*wo2_, target, -xi_plus.at(0).at(1)*inv_sum_eplus);

    // Do it for context j-
    grad_.addVector(hidden_, -xi_minus.at(0).at(0)*inv_sum_eminus);
    grad_.addRow(*wo_, negTarget, xi_minus.at(0).at(0)*inv_sum_eminus);
    // j=1
    grad_.addVector(hidden_, -xi_minus.at(0).at(1)*inv_sum_eminus);
    grad_.addRow(*wo2_, negTarget, xi_minus.at(0).at(1)*inv_sum_eminus);

    // (2) Update grad2_
    grad2_.addVector(hidden2_, xi_plus.at(1).at(0)*inv_sum_eplus);
    grad2_.addRow(*wo_, target, -xi_plus.at(1).at(0)*inv_sum_eplus);
    // j=1
    grad2_.addVector(hidden2_, xi_plus.at(1).at(1)*inv_sum_eplus);
    grad2_.addRow(*wo2_, target, -xi_plus.at(1).at(1)*inv_sum_eplus);

    // Do it for context j-   
    grad2_.addVector(hidden2_, -xi_minus.at(1).at(0)*inv_sum_eminus);
    grad2_.addRow(*wo_, negTarget, xi_minus.at(1).at(0)*inv_sum_eminus);
    // j=1
    grad2_.addVector(hidden2_, -xi_minus.at(1).at(1)*inv_sum_eminus);
    grad2_.addRow(*wo2_, negTarget, xi_minus.at(1).at(1)*inv_sum_eminus);

    ///////////////////////////////
    // (3) Update wo_[target]     --- this involves eplus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_plus.at(0).at(0));
    temp_.addRow(*wo_, target, xi_plus.at(0).at(0));
    // from i=1
    temp_.addVector(hidden2_, -xi_plus.at(1).at(0));
    temp_.addRow(*wo_, target, xi_plus.at(1).at(0));
    wo_->addRow(temp_, target, inv_sum_eplus);
    
    // (4) Update wo2[target]     --- this involves eplus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_plus.at(0).at(1));
    temp_.addRow(*wo2_, target, xi_plus.at(0).at(1));
    // from i=1
    temp_.addVector(hidden2_, -xi_plus.at(1).at(1));
    temp_.addRow(*wo2_, target, xi_plus.at(1).at(1));
    wo2_->addRow(temp_, target, inv_sum_eplus);

    // (5) Update wo_[negTarget]  --- this involves eminus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_minus.at(0).at(0));
    temp_.addRow(*wo_, negTarget, xi_minus.at(0).at(0));
    // from i=1
    temp_.addVector(hidden2_, -xi_minus.at(1).at(0));
    temp_.addRow(*wo_, negTarget, xi_minus.at(1).at(0));
    wo_->addRow(temp_, negTarget, -inv_sum_eminus);
    
    // (6) Update wo2_[negTarget] --- this involves eminus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_minus.at(0).at(1));
    temp_.addRow(*wo2_, negTarget, xi_minus.at(0).at(1));
    // from i=1
    temp_.addVector(hidden2_, -xi_minus.at(1).at(1));
    temp_.addRow(*wo2_, negTarget, xi_minus.at(1).at(1));
    wo2_->addRow(temp_, negTarget, -inv_sum_eminus);
  }
  return std::max((real) 0.0, loss);
}

real Model::negativeSamplingMultiVecExpdot(int32_t target, real lr){
  grad_.zero();
  grad2_.zero();
  // 1. we compute sim1 and sim2 and see if we need to update
  std::vector<float> eplus_result = energy_expdot(target);
  int32_t negTarget = getNegative(target);
  std::vector<float> eminus_result = energy_expdot(negTarget);
  real loss = args_->margin - eplus_result.at(2) + eminus_result.at(2);
  if (loss > 0.0){
    // 2. The goal is to update grad_ and grad2_
    real inv_sum_eplus = lr*(1./eplus_result.at(1))*(-1./args_->var_scale); // plus
    real inv_sum_eminus = lr*(1./eminus_result.at(1))*(-1./args_->var_scale); // minus
    
    std::vector< std::vector<real> > xi_plus;
    std::vector<real> vp1;
    std::vector<real> vp2;
    vp1.push_back(eplus_result.at(3));
    vp1.push_back(eplus_result.at(4));
    vp2.push_back(eplus_result.at(5));
    vp2.push_back(eplus_result.at(6));
    xi_plus.push_back(vp1);
    xi_plus.push_back(vp2);
    
    std::vector< std::vector<real> > xi_minus;
    std::vector<real> vm1;
    std::vector<real> vm2;
    vm1.push_back(eminus_result.at(3));
    vm1.push_back(eminus_result.at(4));
    vm2.push_back(eminus_result.at(5));
    vm2.push_back(eminus_result.at(6));
    xi_minus.push_back(vm1);
    xi_minus.push_back(vm2);

    // (1) Update grad_
    // Do it for context j+
    // j=0
    grad_.addRow(*wo_, target, -xi_plus.at(0).at(0)*inv_sum_eplus);

    // j=1
    grad_.addRow(*wo2_, target, -xi_plus.at(0).at(1)*inv_sum_eplus);

    // Do it for context j-
    grad_.addRow(*wo_, negTarget, xi_minus.at(0).at(0)*inv_sum_eminus);
    // j=1
    grad_.addRow(*wo2_, negTarget, xi_minus.at(0).at(1)*inv_sum_eminus);

    // (2) Update grad2_
    grad2_.addRow(*wo_, target, -xi_plus.at(1).at(0)*inv_sum_eplus);
    // j=1
    grad2_.addRow(*wo2_, target, -xi_plus.at(1).at(1)*inv_sum_eplus);

    // Do it for context j-   
    grad2_.addRow(*wo_, negTarget, xi_minus.at(1).at(0)*inv_sum_eminus);
    // j=1
    grad2_.addRow(*wo2_, negTarget, xi_minus.at(1).at(1)*inv_sum_eminus);

    ///////////////////////////////
    // (3) Update wo_[target]     --- this involves eplus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_plus.at(0).at(0));
    // from i=1
    temp_.addVector(hidden2_, -xi_plus.at(1).at(0));
    wo_->addRow(temp_, target, inv_sum_eplus);
    
    // (4) Update wo2[target]     --- this involves eplus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_plus.at(0).at(1));
    // from i=1
    temp_.addVector(hidden2_, -xi_plus.at(1).at(1));
    wo2_->addRow(temp_, target, inv_sum_eplus);

    // (5) Update wo_[negTarget]  --- this involves eminus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_minus.at(0).at(0));
    // from i=1
    temp_.addVector(hidden2_, -xi_minus.at(1).at(0));
    wo_->addRow(temp_, negTarget, -inv_sum_eminus);
    
    // (6) Update wo2_[negTarget] --- this involves eminus
    temp_.zero();
    // from i=0
    temp_.addVector(hidden_, -xi_minus.at(0).at(1));
    // from i=1
    temp_.addVector(hidden2_, -xi_minus.at(1).at(1));
    wo2_->addRow(temp_, negTarget, -inv_sum_eminus);
  }
  return std::max((real) 0.0, loss);
}

// Feb6 TODO
real Model::negativeSamplingMultiVecVar(int32_t wordidx, int32_t target, real lr){
  grad_.zero();
  grad2_.zero();
  gradvar_.zero();
  gradvar2_.zero();
  // 1. we compute sim1 and sim2 and see if we need to update
  std::vector<float> eplus_result = energy_vecvar(wordidx, target);
  int32_t negTarget = getNegative(target);
  std::vector<float> eminus_result = energy_vecvar(wordidx, negTarget);
  real loss = args_->margin - eplus_result.at(2) + eminus_result.at(2);
  if (loss > 0.0){
    // 2. The goal is to update grad_ and grad2_
    real inv_sum_eplus  = lr*(1./eplus_result.at(1));
    real inv_sum_eminus = lr*(1./eminus_result.at(1));

    std::vector< std::vector<real> > xi_plus;
    std::vector<real> vp1;
    std::vector<real> vp2;
    vp1.push_back(eplus_result.at(3));
    vp1.push_back(eplus_result.at(4));
    vp2.push_back(eplus_result.at(5));
    vp2.push_back(eplus_result.at(6));
    xi_plus.push_back(vp1);
    xi_plus.push_back(vp2);
    
    std::vector< std::vector<real> > xi_minus;
    std::vector<real> vm1;
    std::vector<real> vm2;
    vm1.push_back(eminus_result.at(3));
    vm1.push_back(eminus_result.at(4));
    vm2.push_back(eminus_result.at(5));
    vm2.push_back(eminus_result.at(6));
    xi_minus.push_back(vm1);
    xi_minus.push_back(vm2);

    // with gradients, update gradvar_ and gradvar2_
    if (args_->var){
    // updating gradvar_
    // update for context j+
    // for j=0
    temp_.zero();
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar_->at(target, ii)));
      temp_.data_[ii] += 0.5*inv_sum_eplus*xi_plus.at(0).at(0)*(-invsumd + pow(invsumd, 2.)*pow(hidden_.data_[ii] - wo_->at(target, ii), 2.));
    }
    // for j=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar2_->at(target, ii)));
      temp_.data_[ii] += 0.5*inv_sum_eplus*xi_plus.at(0).at(1)*(-invsumd + pow(invsumd, 2.)*pow(hidden_.data_[ii] - wo2_->at(target, ii), 2.));
    }
    // update for context j-
    // for j=0
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar_->at(negTarget, ii)));
      temp_.data_[ii] += -0.5*inv_sum_eminus*xi_minus.at(0).at(0)*(-invsumd + pow(invsumd, 2.)*pow(hidden_.data_[ii] - wo_->at(negTarget, ii), 2.));
    }
    // for j=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar2_->at(negTarget, ii)));
      temp_.data_[ii] += -0.5*inv_sum_eminus*xi_minus.at(0).at(1)*(-invsumd + pow(invsumd, 2.)*pow(hidden_.data_[ii] - wo2_->at(negTarget, ii), 2.));
    }
    // in the end, multiple with d_i to do derivative against the log instead
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      gradvar_.data_[ii] = exp(invar_->at(wordidx, ii))*temp_.data_[ii];
    }
    // updating gradvar2_
    // update for context j+
    // for j=0
    temp_.zero();
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar_->at(target, ii)));
      temp_.data_[ii] += 0.5*inv_sum_eplus*xi_plus.at(1).at(0)*(-invsumd + pow(invsumd, 2.)*pow(hidden2_.data_[ii] - wo_->at(target, ii), 2.));
    }
    // for j=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar2_->at(target, ii)));
      temp_.data_[ii] += 0.5*inv_sum_eplus*xi_plus.at(1).at(1)*(-invsumd + pow(invsumd, 2.)*pow(hidden2_.data_[ii] - wo2_->at(target, ii), 2.));
    }
    // update for context j-
    // for j=0
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar_->at(negTarget, ii)));
      temp_.data_[ii] += -0.5*inv_sum_eminus*xi_minus.at(1).at(0)*(-invsumd + pow(invsumd, 2.)*pow(hidden2_.data_[ii] - wo_->at(negTarget, ii), 2.));
    }
    // for j=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar2_->at(negTarget, ii)));
      temp_.data_[ii] += -0.5*inv_sum_eminus*xi_minus.at(1).at(1)*(-invsumd + pow(invsumd, 2.)*pow(hidden2_.data_[ii] - wo2_->at(negTarget, ii), 2.));
    }
    // in the end, multiple with d_i to do derivative against the log instead
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      gradvar2_.data_[ii] = exp(invar2_->at(wordidx, ii))*temp_.data_[ii];
    }

    // update outvar_[target]
    temp_.zero();
    // from i=0
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar_->at(target, ii)));
      temp_.data_[ii] += -0.5*inv_sum_eplus*xi_plus.at(0).at(0)*(-invsumd + pow(invsumd, 2.)*pow(hidden_.data_[ii] - wo_->at(target, ii), 2.));
    }
    // from i=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar_->at(target, ii)));
      temp_.data_[ii] += -0.5*inv_sum_eplus*xi_plus.at(1).at(0)*(-invsumd + pow(invsumd, 2.)*pow(hidden2_.data_[ii] - wo_->at(target, ii), 2.));
    }
    temp_.mulExpRow(*outvar_, target); // make it a derivative against log
    outvar_->addRow(temp_, target, 1.);

    // update outvar2_[target]
    temp_.zero();
    // from i=0
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar2_->at(target, ii)));
      temp_.data_[ii] += -0.5*inv_sum_eplus*xi_plus.at(0).at(1)*(-invsumd + pow(invsumd, 2.)*pow(hidden_.data_[ii] - wo2_->at(target, ii), 2.));
    }
    // from i=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar2_->at(target, ii)));
      temp_.data_[ii] += -0.5*inv_sum_eplus*xi_plus.at(1).at(1)*(-invsumd + pow(invsumd, 2.)*pow(hidden2_.data_[ii] - wo2_->at(target, ii), 2.));
    }
    temp_.mulExpRow(*outvar2_, target);
    outvar2_->addRow(temp_, target, 1.);

    // update outvar_[negTarget]
    // the loss has different sign (compared to target)
    temp_.zero();
    // from i=0
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar_->at(negTarget, ii)));
      temp_.data_[ii] += 0.5*inv_sum_eplus*xi_plus.at(0).at(0)*(-invsumd + pow(invsumd, 2.)*pow(hidden_.data_[ii] - wo_->at(negTarget, ii), 2.));
    }
    // from i=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar_->at(negTarget, ii)));
      temp_.data_[ii] += 0.5*inv_sum_eplus*xi_plus.at(1).at(0)*(-invsumd + pow(invsumd, 2.)*pow(hidden2_.data_[ii] - wo_->at(negTarget, ii), 2.));
    }
    temp_.mulExpRow(*outvar_, negTarget);
    outvar_->addRow(temp_, negTarget, 1.);

    // update outvar2_[negTarget]
    temp_.zero();
    // from i=0
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar2_->at(negTarget, ii)));
      temp_.data_[ii] += 0.5*inv_sum_eplus*xi_plus.at(0).at(1)*(-invsumd + pow(invsumd, 2.)*pow(hidden_.data_[ii] - wo2_->at(negTarget, ii), 2.));
    }
    // from i=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar2_->at(negTarget, ii)));
      temp_.data_[ii] += 0.5*inv_sum_eplus*xi_plus.at(1).at(1)*(-invsumd + pow(invsumd, 2.)*pow(hidden2_.data_[ii] - wo2_->at(negTarget, ii), 2.));
    }
    temp_.mulExpRow(*outvar2_, negTarget);
    outvar2_->addRow(temp_, negTarget, 1.);
    }

    // (1) Update grad_
    // Do it for context j+
    // j=0
    for (int64_t ii = 0; ii < grad_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar_->at(target, ii)));
      grad_.data_[ii] += inv_sum_eplus*xi_plus.at(0).at(0)*(-invsumd*(hidden_.data_[ii] - wo_->at(target, ii)));
    }
    // j=1
    for (int64_t ii = 0; ii < grad_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar2_->at(target, ii)));
      grad_.data_[ii] += inv_sum_eplus*xi_plus.at(0).at(1)*(-invsumd*(hidden_.data_[ii] - wo2_->at(target, ii)));
    }

    // Do it for context j-
    for (int64_t ii = 0; ii < grad_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar_->at(negTarget, ii)));
      grad_.data_[ii] += -inv_sum_eminus*xi_minus.at(0).at(0)*(-invsumd*(hidden_.data_[ii] - wo_->at(negTarget, ii)));
    }
    
    // j=1
    for (int64_t ii = 0; ii < grad_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar2_->at(negTarget, ii)));
      grad_.data_[ii] += -inv_sum_eminus*xi_minus.at(0).at(1)*(-invsumd*(hidden_.data_[ii] - wo2_->at(negTarget, ii)));
    }

    // (2) Update grad2_
    for (int64_t ii = 0; ii < grad2_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar_->at(target, ii)));
      grad2_.data_[ii] += inv_sum_eplus*xi_plus.at(1).at(0)*(-invsumd*(hidden2_.data_[ii] - wo_->at(target, ii)));
    }
    // j=1
    for (int64_t ii = 0; ii < grad2_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar2_->at(target, ii)));
      grad2_.data_[ii] += inv_sum_eplus*xi_plus.at(1).at(1)*(-invsumd*(hidden2_.data_[ii] - wo2_->at(target, ii)));
    }

    // Do it for context j-   
    for (int64_t ii = 0; ii < grad2_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar_->at(negTarget, ii)));
      grad2_.data_[ii] += -inv_sum_eminus*xi_minus.at(1).at(0)*(-invsumd*(hidden2_.data_[ii] - wo_->at(negTarget, ii)));
    }

    // j=1
    for (int64_t ii = 0; ii < grad2_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar2_->at(negTarget, ii)));
      grad2_.data_[ii] += -inv_sum_eminus*xi_minus.at(1).at(1)*(-invsumd*(hidden2_.data_[ii] - wo2_->at(negTarget, ii)));
    }

    ///////////////////////////////
    // (3) Update wo_[target]     --- this involves eplus
    temp_.zero();
    // from i=0
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar_->at(target, ii)));
      temp_[ii] += xi_plus.at(0).at(0)*invsumd*(hidden_.data_[ii] - wo_->at(target, ii));
    }
    // from i=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar_->at(target, ii)));
      temp_[ii] += xi_plus.at(1).at(0)*invsumd*(hidden2_.data_[ii] - wo_->at(target, ii));
    }
    wo_->addRow(temp_, target, inv_sum_eplus);
    
    // (4) Update wo2[target]     --- this involves eplus
    temp_.zero();
    // from i=0
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar2_->at(target, ii)));
      temp_[ii] += xi_plus.at(0).at(1)*invsumd*(hidden_.data_[ii] - wo2_->at(target, ii));
    }
    // from i=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar2_->at(target, ii)));
      temp_[ii] += xi_plus.at(1).at(1)*invsumd*(hidden2_.data_[ii] - wo2_->at(target, ii));
    }
    wo2_->addRow(temp_, target, inv_sum_eplus);

    // (5) Update wo_[negTarget]  --- this involves eminus
    temp_.zero();
    // from i=0
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar_->at(negTarget, ii)));
      temp_[ii] += xi_minus.at(0).at(0)*invsumd*(hidden_.data_[ii] - wo_->at(negTarget, ii));
    }
    // from i=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar_->at(negTarget, ii)));
      temp_[ii] += xi_minus.at(1).at(0)*invsumd*(hidden2_.data_[ii] - wo_->at(negTarget, ii));
    }
    wo_->addRow(temp_, negTarget, -inv_sum_eminus);
    
    // (6) Update wo2_[negTarget] --- this involves eminus
    temp_.zero();
    // from i=0
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar_->at(wordidx, ii)) + exp(outvar2_->at(negTarget, ii)));
      temp_[ii] += xi_minus.at(0).at(1)*invsumd*(hidden_.data_[ii] - wo2_->at(negTarget, ii));
    }
    // from i=1
    for (int64_t ii = 0; ii < temp_.m_; ii++) {
      real invsumd = 1./(1e-8 + exp(invar2_->at(wordidx, ii)) + exp(outvar2_->at(negTarget, ii)));
      temp_[ii] += xi_minus.at(1).at(1)*invsumd*(hidden2_.data_[ii] - wo2_->at(negTarget, ii));
    }
    wo2_->addRow(temp_, negTarget, -inv_sum_eminus);
  }
  return std::max((real) 0.0, loss);
}

real Model::hierarchicalSoftmax(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  const std::vector<bool>& binaryCode = codes[target];
  const std::vector<int32_t>& pathToRoot = paths[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
  }
  return loss;
}

void Model::computeOutputSoftmax(Vector& hidden, Vector& output) const {
  if (quant_ && args_->qout) {
    output.mul(*qwo_, hidden);
  } else {
    output.mul(*wo_, hidden);
  }
  real max = output[0], z = 0.0;
  for (int32_t i = 0; i < osz_; i++) {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] /= z;
  }
}

void Model::computeOutputSoftmax() {
  computeOutputSoftmax(hidden_, output_);
}

real Model::softmax(int32_t target, real lr) {
  grad_.zero();
  computeOutputSoftmax();
  for (int32_t i = 0; i < osz_; i++) {
    real label = (i == target) ? 1.0 : 0.0;
    real alpha = lr * (label - output_[i]);
    grad_.addRow(*wo_, i, alpha);
    wo_->addRow(hidden_, i, alpha);
  }
  return -log(output_[target]);
}


void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden) const {
  // by default, no dropout
  computeHidden(input, hidden, false, false);
}



void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden, bool dropout_dict, bool dropout_sub) const {
  assert(hidden.size() == hsz_);
  hidden.zero();
  int jjj = 0;
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    if(quant_) {
      hidden.addRow(*qwi_, *it);
    } else {
      if (!(args_->include_dictemb) && (jjj == 0)){
        // if include_dictemb is false, then also do the adding
        // if jjj != 0, do the adding (later)
      } else {
        if (!dropout_sub){
          hidden.addRow(*wi_, *it);
        }
      }
    }
    jjj++;
  }
  if (!(args_->include_dictemb) and input.size() > 1) {
    // make sure we're not dividing by zero
    hidden.mul(1.0 / (input.size() - 1));
  } else {
    hidden.mul(1.0 / input.size());
  }

  // if adding dictemb outside, add the first element of input (ngrams)
  if (args_->add_dictemb){
    for (auto it = input.cbegin(); it!= input.cend(); ++it){
      hidden.addRow(*wi_, *it);
      break;
    }
  }
}

void Model::computeHidden2(const std::vector<int32_t>& input, Vector& hidden, bool dropout_dict, bool dropout_sub) const {
  assert(hidden.size() == hsz_);
  hidden.zero();
  int jjj = 0;
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    if(quant_) {
      hidden.addRow(*qwi_, *it);
    } else {
      if (!(args_->include_dictemb) && (jjj == 0)){
        // if include_dictemb is false, then also do the adding
        // if jjj != 0, do the adding (later)
      } else {
        if (!dropout_sub){
          hidden.addRow(*wi2_, *it);
        }
      }
    }
    jjj++;
  }
  if (!(args_->include_dictemb) and input.size() > 1) {
    // make sure we're not dividing by zero
    hidden.mul(1.0 / (input.size() - 1));
  } else {
    hidden.mul(1.0 / input.size());
  }

  // if adding dictemb outside, add the first element of input (ngrams)
  if (args_->add_dictemb){
    for (auto it = input.cbegin(); it!= input.cend(); ++it){
      hidden.addRow(*wi2_, *it);
      break;
    }
  }
}

void Model::computeHidden2_mv(const std::vector<int32_t>& input, Vector& hidden) const {
  // Just using the vector for component 2
  assert(hidden.size() == hsz_);
  hidden.zero();
  int jjj = 0;
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi2_, *it);
    break;
  }
}

bool Model::comparePairs(const std::pair<real, int32_t> &l,
                         const std::pair<real, int32_t> &r) {
  return l.first > r.first;
}

void Model::predict(const std::vector<int32_t>& input, int32_t k,
                    std::vector<std::pair<real, int32_t>>& heap,
                    Vector& hidden, Vector& output) const {
  assert(k > 0);
  heap.reserve(k + 1);
  computeHidden(input, hidden);
  if (args_->loss == loss_name::hs) {
    dfs(k, 2 * osz_ - 2, 0.0, heap, hidden);
  } else {
    findKBest(k, heap, hidden, output);
  }
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void Model::predict(const std::vector<int32_t>& input, int32_t k,
                    std::vector<std::pair<real, int32_t>>& heap) {
  predict(input, k, heap, hidden_, output_);
}

void Model::findKBest(int32_t k, std::vector<std::pair<real, int32_t>>& heap,
                      Vector& hidden, Vector& output) const {
  computeOutputSoftmax(hidden, output);
  for (int32_t i = 0; i < osz_; i++) {
    if (heap.size() == k && log(output[i]) < heap.front().first) {
      continue;
    }
    heap.push_back(std::make_pair(log(output[i]), i));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}

void Model::dfs(int32_t k, int32_t node, real score,
                std::vector<std::pair<real, int32_t>>& heap,
                Vector& hidden) const {
  if (heap.size() == k && score < heap.front().first) {
    return;
  }

  if (tree[node].left == -1 && tree[node].right == -1) {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  real f;
  if (quant_ && args_->qout) {
    f= sigmoid(qwo_->dotRow(hidden, node - osz_));
  } else {
    f= sigmoid(wo_->dotRow(hidden, node - osz_));
  }

  dfs(k, tree[node].left, score + log(1.0 - f), heap, hidden);
  dfs(k, tree[node].right, score + log(f), heap, hidden);
}

float probRand() {
    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<int> distribution(0,1000);
    return distribution(generator)/(1.*1000);
}

void Model::update(const std::vector<int32_t>& input, int32_t target, real lr) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0) return;

  // get the word index --> this is the first element in 'input'
  int32_t wordidx = 0;
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wordidx = *it;
    break;
  }
  
  computeHidden(input, hidden_, false, false);
  computeHidden2_mv(input, hidden2_);
  if (args_->loss == loss_name::ns) {
    if (args_->multi){
      if (args_->var) {
        loss_ += negativeSamplingMultiVecVar(wordidx, target, lr);
      } else{
      if (args_->expdot) {
        loss_ += negativeSamplingMultiVecExpdot(target, lr);
      } else {
        loss_ += negativeSamplingMultiVec2(target, lr);
      }
      }
    } else {
      if (args_->var) {
        // not using this version
      } else {
      if (args_->expdot) {
        loss_ += negativeSamplingSingleExpdot(target, lr);
      } else {
        loss_ += negativeSampling(target, lr);
      }
      }
    }
  } else if (args_->loss == loss_name::hs) {
    // not using
    loss_ += hierarchicalSoftmax(target, lr);
  } else {
    // not using
    loss_ += softmax(target, lr);
  }
  nexamples_ += 1;

  // not using
  if (args_->model == model_name::sup) {
    grad_.mul(1.0 / input.size());
  }

  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addRow(grad_, *it, 1.0);
  }

  // MV mode - use only vector representation for cluster 2
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi2_->addRow(grad2_, *it, 1.0);
    break;
  }
  // update var
  if (args_->var){
    invar_->addRow(gradvar_, wordidx, 1.0);
    invar2_->addRow(gradvar2_, wordidx, 1.0);
  }
}

void Model::groupSparsityRegularization(int min, int max, int num_gs_samples, double strength){
  // sampling from the uniform interval [min, max)
  grad_.zero();
  std::uniform_int_distribution<> uniform(min, max-1);
  if (args_->gs_lambda > 1e-12){
    for (int ii = 0; ii < num_gs_samples; ii++){
      // Note: osz_ is the number of words in the dictionary
      // Perhaps adjusts the distribution of this sampler
      int32_t idx = uniform(rng);
      loss_ += groupSparsityRegularization(strength, idx);
    }
  }
}

real Model::groupSparsityRegularization(double reg_strength, int32_t word){
  // To be efficient, only do it if the strength is non-zero
  if (reg_strength > 0.0000000001) {
    real norm = wi_->l2NormRow(word);
    real loss = reg_strength*norm;
    // 2. update the wi_ accordingly based on the gradient
    // note: reuse the grad variable here
    grad_.zero();
    grad_.addRow(*wi_, word, -reg_strength/(norm + 0.00001));
    wi_->addRow(grad_, word, 1.0);
    return loss;
  } else {
    return 0.0;
  }
}

void Model::setTargetCounts(const std::vector<int64_t>& counts) {
  assert(counts.size() == osz_);
  if (args_->loss == loss_name::ns) {
    initTableNegatives(counts);
  }
  if (args_->loss == loss_name::hs) {
    buildTree(counts);
  }
}

void Model::initTableNegatives(const std::vector<int64_t>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    real c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives.push_back(i);
    }
  }
  std::shuffle(negatives.begin(), negatives.end(), rng);
}

int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives[negpos];
    negpos = (negpos + 1) % negatives.size();
  } while (target == negative);
  return negative;
}

void Model::buildTree(const std::vector<int64_t>& counts) {
  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree[i].parent = -1;
    tree[i].left = -1;
    tree[i].right = -1;
    tree[i].count = 1e15;
    tree[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2];
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree[leaf].count < tree[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    tree[i].left = mini[0];
    tree[i].right = mini[1];
    tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
    tree[mini[0]].parent = i;
    tree[mini[1]].parent = i;
    tree[mini[1]].binary = true;
  }
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree[j].parent != -1) {
      path.push_back(tree[j].parent - osz_);
      code.push_back(tree[j].binary);
      j = tree[j].parent;
    }
    paths.push_back(path);
    codes.push_back(code);
  }
}

real Model::getLoss() const {
  return loss_ / nexamples_;
}

void Model::initSigmoid() {
  t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
  }
}

void Model::initLog() {
  t_log = new real[LOG_TABLE_SIZE + 1];
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log[i] = std::log(x);
  }
}

real Model::log(real x) const {
  if (x > 1.0) {
    return 0.0;
  }
  int i = int(x * LOG_TABLE_SIZE);
  return t_log[i];
}

real Model::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid[i];
  }
}

}
