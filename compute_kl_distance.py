import numpy as np 
import pickle
import sys
import scipy.special

import model_config
from run import make_approximate
import tensorflow as tf 
import argparse


parser = argparse.ArgumentParser(description='Compute KL distance between approximate predictive distributions.')

parser.add_argument('--folder', help='location to read the training result',
                    required=False,
                    type=str,
                    default='result')

parser.add_argument('--outfolder', help='location to write the data',
                    required=False,
                    type=str,
                    default='plot_data')

parser.add_argument('--exper', help='in {moon, moon_random, moon_rm_30, moon_rm_40, moon_rm_50, banknote_authentication1}',
                    required=False,
                    type=str,
                    default='moon')

parser.add_argument('--appr', help='in {gauss_fullcov, gauss_diag, maf}',
                    required=False,
                    type=str,
                    default='gauss_fullcov')

args = parser.parse_args()

folder = args.folder
outfolder = args.outfolder
experiment = args.exper
approximate = args.appr
nbijector = 15
nhidden = 5

if experiment.startswith("moon"):
    eubo_mode_percentages = [0.5, 0.1, 1e-3, 1e-5, 1e-9, 0.0]
    elbo_mode_percentages = [0.5, 0.1, 1e-3, 1e-5, 1e-9, 0.0]

elif experiment == "banknote_authentication1":
    if approximate == "gauss_fullcov":
        eubo_mode_percentages = [.5, .1, 1e-3, 1e-5, 1e-9, 0.0]
        elbo_mode_percentages = [.5, .1, 1e-3, 1e-5, 1e-9, 0.0]
    elif approximate == "maf":
        eubo_mode_percentages = [1e-3, 1e-5, 1e-7, 1e-9, 0.0]
        elbo_mode_percentages = [1e-3, 1e-5, 1e-7, 1e-9, 0.0]

else:
    raise Exception("Unknown experiment {}".format(experiment))



model_data = model_config.get_experiment(experiment)
# dim, nparam, model, data, removed_data, remain_data

n_param_samples = 100
param_sample_batch = 100
n_iter = int(n_param_samples / param_sample_batch)

approx_dist, approx_config = make_approximate(approximate, nbijector, nhidden)
approx_model = approx_dist(model_data['nparam'], **approx_config)

param_samples = approx_model.sample(param_sample_batch)
removed_likelihoods = model_data['model'].log_posterior_predictive(param_samples, model_data['removed_data'].astype(np.float64))
# (nclass,ndata,nsample)

remain_likelihoods = model_data['model'].log_posterior_predictive(param_samples, model_data['remain_data'].astype(np.float64))
# (nclass,ndata,nsample)

def get_likelihoods(learned_param):

    with tf.Session() as sess:
        approx_model.load_param(learned_param, sess)
        
        removed_likelihoods_np = []
        remain_likelihoods_np = []

        for _ in range(n_iter):
            removed_likelihoods_np_i, remain_likelihoods_np_i = sess.run([removed_likelihoods, remain_likelihoods])

            removed_likelihoods_np_i = scipy.special.logsumexp(removed_likelihoods_np_i, axis=2, keepdims=True) - np.log(removed_likelihoods_np_i.shape[2])
            # (nclass, ndata, 1)

            remain_likelihoods_np_i = scipy.special.logsumexp(remain_likelihoods_np_i, axis=2, keepdims=True) - np.log(remain_likelihoods_np_i.shape[2])
            # (nclass, ndata, 1)

            removed_likelihoods_np.append(removed_likelihoods_np_i)
            remain_likelihoods_np.append(remain_likelihoods_np_i)
        
        removed_likelihoods_np = np.concatenate(removed_likelihoods_np, axis=2) # concat over the sample dimension
        remain_likelihoods_np = np.concatenate(remain_likelihoods_np, axis=2)
        
        removed_likelihoods_np = scipy.special.logsumexp(removed_likelihoods_np, axis=2) - np.log(removed_likelihoods_np.shape[2])
        remain_likelihoods_np = scipy.special.logsumexp(remain_likelihoods_np, axis=2) - np.log(remain_likelihoods_np.shape[2])

    return removed_likelihoods_np, remain_likelihoods_np
    # (nclass, ndata)

print("Likelihood of full-data posterior"); sys.stdout.flush()
full_learned_param = pickle.load(
    open("{}/{}/{}/full_data_post.p".format(folder, experiment, approximate), "rb"))
full_removed_likelihood_np, full_remain_likelihood_np = get_likelihoods(full_learned_param)

print("Likelihood of re-trained posterior"); sys.stdout.flush()
retrain_learned_param = pickle.load(
    open("{}/{}/{}/remain_data_retrain_post.p".format(folder, experiment, approximate), "rb"))
retrain_removed_likelihood_np, retrain_remain_likelihood_np = get_likelihoods(retrain_learned_param)


def cal_likelihood_diff(ll1, ll2):
    # compute the KL divergence
    # ll1: (nclass, ndata)
    # ll2: (nclass, ndata)
    # expectation_over_exp(ll2) (ll2 - ll1)
    probs = np.exp(ll2)
    kl_distances = np.sum(probs * (ll2 - ll1), axis = 0)
    # (ndata,)
    return np.mean(kl_distances), np.std(kl_distances)


mean_removed_ll_full_vs_retrain, std_removed_ll_full_vs_retrain = cal_likelihood_diff(
                        full_removed_likelihood_np, retrain_removed_likelihood_np)

removed_likelihood_diffs = {
    "full_vs_retrain": mean_removed_ll_full_vs_retrain,
    "elbo": {
        "vs_retrain": np.zeros(len(elbo_mode_percentages)),
        "vs_full": np.zeros(len(elbo_mode_percentages))
    },
    "eubo": {
        "vs_retrain": np.zeros(len(elbo_mode_percentages)),
        "vs_full": np.zeros(len(elbo_mode_percentages))
    }
}
std_removed_likelihood_diffs = {
    "full_vs_retrain": std_removed_ll_full_vs_retrain,
    "elbo": {
        "vs_retrain": np.zeros(len(elbo_mode_percentages)),
        "vs_full": np.zeros(len(elbo_mode_percentages))
    },
    "eubo": {
        "vs_retrain": np.zeros(len(elbo_mode_percentages)),
        "vs_full": np.zeros(len(elbo_mode_percentages))
    }
}


mean_remain_ll_full_vs_retrain, std_remain_ll_full_vs_retrain = cal_likelihood_diff(
                        full_remain_likelihood_np, retrain_remain_likelihood_np)

remain_likelihood_diffs = {
    "full_vs_retrain": mean_remain_ll_full_vs_retrain,
    "elbo": {
        "vs_retrain": np.zeros(len(elbo_mode_percentages)),
        "vs_full": np.zeros(len(elbo_mode_percentages))
    },
    "eubo": {
        "vs_retrain": np.zeros(len(elbo_mode_percentages)),
        "vs_full": np.zeros(len(elbo_mode_percentages))
    }
}
std_remain_likelihood_diffs = {
    "full_vs_retrain": std_remain_ll_full_vs_retrain,
    "elbo": {
        "vs_retrain": np.zeros(len(elbo_mode_percentages)),
        "vs_full": np.zeros(len(elbo_mode_percentages))
    },
    "eubo": {
        "vs_retrain": np.zeros(len(elbo_mode_percentages)),
        "vs_full": np.zeros(len(elbo_mode_percentages))
    }
}


for i,percentage in enumerate(elbo_mode_percentages):
    print("Likelihood of ELBO unlearned posterior {}".format(percentage)); sys.stdout.flush()
    learned_param = pickle.load(
        open("{}/{}/{}/data_remain_data_by_unlearn_elbo_{}.p".format(
            folder,experiment, approximate, percentage), "rb"))
    removed_likelihood_i, remain_likelihood_i = get_likelihoods(learned_param)

    removed_likelihood_diffs["elbo"]["vs_retrain"][i], std_removed_likelihood_diffs["elbo"]["vs_retrain"][i] = \
                        cal_likelihood_diff(
                        removed_likelihood_i, retrain_removed_likelihood_np)
    remain_likelihood_diffs["elbo"]["vs_retrain"][i], std_remain_likelihood_diffs["elbo"]["vs_retrain"][i] = \
                        cal_likelihood_diff(
                        remain_likelihood_i, retrain_remain_likelihood_np)

    removed_likelihood_diffs["elbo"]["vs_full"][i], std_removed_likelihood_diffs["elbo"]["vs_full"][i] = \
                        cal_likelihood_diff(
                        removed_likelihood_i, full_removed_likelihood_np)
    remain_likelihood_diffs["elbo"]["vs_full"][i], std_remain_likelihood_diffs["elbo"]["vs_full"][i] = \
                        cal_likelihood_diff(
                        remain_likelihood_i, full_remain_likelihood_np)

    
for i,percentage in enumerate(eubo_mode_percentages):
    print("Likelihood of EUBO unlearned posterior {}".format(percentage)); sys.stdout.flush()
    learned_param = pickle.load(open("{}/{}/{}/data_remain_data_by_unlearn_eubo_{}.p".format(folder,
        experiment, approximate, percentage), "rb"))
    removed_likelihood_i, remain_likelihood_i = get_likelihoods(learned_param)

    removed_likelihood_diffs["eubo"]["vs_retrain"][i], std_removed_likelihood_diffs["eubo"]["vs_retrain"][i] = \
                        cal_likelihood_diff(
                        removed_likelihood_i, retrain_removed_likelihood_np)
    remain_likelihood_diffs["eubo"]["vs_retrain"][i], std_remain_likelihood_diffs["eubo"]["vs_retrain"][i] = \
                        cal_likelihood_diff(
                        remain_likelihood_i, retrain_remain_likelihood_np)

    removed_likelihood_diffs["eubo"]["vs_full"][i], std_removed_likelihood_diffs["eubo"]["vs_full"][i] = \
                        cal_likelihood_diff(
                        removed_likelihood_i, full_removed_likelihood_np)
    remain_likelihood_diffs["eubo"]["vs_full"][i], std_remain_likelihood_diffs["eubo"]["vs_full"][i] = \
                        cal_likelihood_diff(
                        remain_likelihood_i, full_remain_likelihood_np)

outfilename = "{}/likelihood_diff_{}_{}.p".format(outfolder, experiment, approximate)

pickle.dump({
    "elbo_mode_percentage": elbo_mode_percentages,
    "eubo_mode_percentage": eubo_mode_percentages,
    "removed_likelihood_diff": removed_likelihood_diffs,
    "remain_likelihood_diff": remain_likelihood_diffs,
    "std_removed_likelihood_diff": std_removed_likelihood_diffs,
    "std_remain_likelihood_diff": std_remain_likelihood_diffs
    },
    open(outfilename, "wb"))
