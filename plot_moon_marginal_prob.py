import numpy as np 
import pickle
import tensorflow as tf 
import scipy.special 

import matplotlib

from matplotlib import rc
rc('text', usetex=True)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


import matplotlib.pyplot as plt 
import seaborn 
plt.style.use('ggplot')

plt.rcParams['lines.linewidth']=1.5
plt.rcParams['axes.facecolor']='w'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import model_config
from run import make_approximate


def num2str(num):
    return str(num).replace('.', '_')
    

import argparse


parser = argparse.ArgumentParser(description='Plot the data, posterior mean and variance of the synthetic moon classification dataset.')

parser.add_argument('--exper', help='in {moon, moon_random, moon_rm_30, moon_rm_40, moon_rm_50}',
                    required=False,
                    type=str,
                    default='moon')


args = parser.parse_args()


nbijector = 15
nhidden = 5

experiment = args.exper
approximate = "gauss_fullcov"
show_cbar = True


approximate_dist, approximate_config = make_approximate(approximate, nbijector, nhidden)
experiment_data = model_config.get_experiment(experiment)

dim = experiment_data['dim']
nparam = experiment_data['nparam']
model = experiment_data['model']
data = experiment_data['data']
remain_data = experiment_data['remain_data']
removed_data = experiment_data['removed_data']
ndata = data.shape[0]

prefix = "result/{}/{}".format(experiment, approximate)
selected_percentages = [1e-5,1e-9, 0.0]

n = 50
plot_xmin = -1.5
plot_xmax = 2.5

x1d = np.linspace(plot_xmin, plot_xmax, n)
x1,x2 = np.meshgrid(x1d, x1d)
x = np.stack([x1.flatten(), x2.flatten()]).T



def padding_dummy_output(data_x):
    # data_x: (n,dim-1)
    # append zero class label to the last column
    return np.concatenate([data_x, np.zeros([data_x.shape[0],1])], axis=1)

n_param_samples = 100
param_sample_batch = 100
n_iter = int(n_param_samples / param_sample_batch)

approx_dist, approx_config = make_approximate(approximate, nbijector, nhidden=5)
approx_model = approx_dist(experiment_data['nparam'], **approx_config)

param_samples = approx_model.sample(param_sample_batch)
posterior_lprobs = experiment_data['model'].log_posterior_predictive(param_samples, padding_dummy_output(x).astype(np.float64))
# (nclass,ndata,nsample)


def get_posterior_lprobs(learned_param):
    with tf.Session() as sess:
        approx_model.load_param(learned_param, sess)
        
        posterior_lprobs_np = []

        for _ in range(n_iter):
            posterior_lprobs_i = sess.run(posterior_lprobs)

            posterior_lprobs_i = scipy.special.logsumexp(posterior_lprobs_i, axis=2, keepdims=True) - np.log(posterior_lprobs_i.shape[2])
            # (nclass, ndata, 1)

            posterior_lprobs_np.append(posterior_lprobs_i)
        
        posterior_lprobs_np = np.concatenate(posterior_lprobs_np, axis=2) # concat over the sample dimension
        # (nclass, ndata, n_param_samples)
        
        posterior_lprobs_np = scipy.special.logsumexp(posterior_lprobs_np, axis=2) - np.log(posterior_lprobs_np.shape[2])
    return posterior_lprobs_np
    # (nclass, ndata)




full_params = pickle.load(open("{}/full_data_post.p".format(prefix), "rb"))
remain_params = pickle.load(open("{}/remain_data_retrain_post.p".format(prefix), "rb"))

full_probs = np.exp(get_posterior_lprobs(full_params))
remain_probs = np.exp(get_posterior_lprobs(remain_params))

elbo_probs = {}
eubo_probs = {}

for percentage in selected_percentages:
    elbo_params = pickle.load(open("{}/data_remain_data_by_unlearn_elbo_{}.p".format(prefix, percentage), "rb"))
    elbo_probs[percentage] = np.exp(get_posterior_lprobs(elbo_params))
    
    eubo_params = pickle.load(open("{}/data_remain_data_by_unlearn_eubo_{}.p".format(prefix, percentage), "rb"))
    eubo_probs[percentage] = np.exp(get_posterior_lprobs(eubo_params))


figsize = (2.4*4, 2.*2)

fig, axs = plt.subplots(2,4,figsize=figsize, tight_layout=True)
hm = seaborn.heatmap(np.flipud(full_probs[1,:].reshape(n,n)), vmin=0, vmax=1, cmap="YlGnBu", ax=axs[0,0], cbar=show_cbar)
hm.set(xticks=list(range(0,50,16)))
hm.set(xticklabels=["{:.1f}".format(i) for i in x1d[list(range(0,50,16))]])
hm.set(yticks=list(range(0,50,16)))
hm.set(yticklabels=["{:.1f}".format(i) for i in x1d[list(reversed(range(0,50,16)))]])
axs[0,0].grid(False)

axs[0,0].set_title(r"$q(y_x|\mathcal{D})$")


hm = seaborn.heatmap(np.flipud(remain_probs[1,:].reshape(n,n)), vmin=0, vmax=1, cmap="YlGnBu", ax=axs[0,1], cbar=show_cbar)
hm.set(xticks=list(range(0,50,16)))
hm.set(xticklabels=["{:.1f}".format(i) for i in x1d[list(range(0,50,16))]])
hm.set(yticks=list(range(0,50,16)))
hm.set(yticklabels=["{:.1f}".format(i) for i in x1d[list(reversed(range(0,50,16)))]])
axs[0,1].grid(False)

axs[0,1].set_title(r"$q(y_x|\mathcal{D}_r)$")

plot_idx = 2

for percentage in selected_percentages:
    axs[int(plot_idx/4), plot_idx%4].grid(False)
    hm = seaborn.heatmap(np.flipud(elbo_probs[percentage][1,:].reshape(n,n)), vmin=0, vmax=1, cmap="YlGnBu", ax=axs[int(plot_idx/4), plot_idx%4], cbar=show_cbar)
    hm.set(xticks=list(range(0,50,16)))
    hm.set(xticklabels=["{:.1f}".format(i) for i in x1d[list(range(0,50,16))]])
    hm.set(yticks=list(range(0,50,16)))
    hm.set(yticklabels=["{:.1f}".format(i) for i in x1d[list(reversed(range(0,50,16)))]])

    axs[int(plot_idx/4), plot_idx%4].set_title(r'$\tilde{q}_v(y_x|\mathcal{D}_r;$' + "$\lambda$={}".format(percentage) + r"$)$")

    plot_idx += 1

    hm = seaborn.heatmap(np.flipud(eubo_probs[percentage][1,:].reshape(n,n)), vmin=0, vmax=1, cmap="YlGnBu", ax=axs[int(plot_idx/4), plot_idx%4], cbar=show_cbar)
    hm.set(xticks=list(range(0,50,16)))
    hm.set(xticklabels=["{:.1f}".format(i) for i in x1d[list(range(0,50,16))]])
    hm.set(yticks=list(range(0,50,16)))
    hm.set(yticklabels=["{:.1f}".format(i) for i in x1d[list(reversed(range(0,50,16)))]])
    axs[int(plot_idx/4), plot_idx%4].grid(False)

    axs[int(plot_idx/4), plot_idx%4].set_title(r'$\tilde{q}_u(y_x|\mathcal{D}_r;$' + "$\lambda$={}".format(percentage) + r"$)$") 

    plot_idx += 1

plt.show()
