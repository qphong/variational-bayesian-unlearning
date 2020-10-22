import tensorflow as tf 
import tensorflow_probability as tfp 
import numpy as np
import scipy.special
import time 
import sys


class Unlearnizable():

    def __init__(self, xdim, nparam, 
                approximate_dist_class,
                approximate_dist_config,
                log_likelihood_func, 
                log_prior_func, 
                nsample=1000):
        """
        log_likelihood_func(theta, data) returns the log_likelihood
            theta: (nsample,nparam)
            data: (ndata,d)
            return (nsample,)
        log_prior_func(theta) returns the log prior probability
            theta: (nsample, nparam)
            return (nsample,)
        """
        self.nparam = nparam
        self.xdim = xdim 
        self.log_likelihood_func = log_likelihood_func 
        self.log_prior_func = log_prior_func 


        self.posterior_dist = approximate_dist_class(self.nparam, **approximate_dist_config)

        self.data_plc = tf.placeholder(shape=(None,self.xdim), dtype=tf.float64)

        post_theta_samples = self.posterior_dist.sample(nsample)
        # (nsample, nparam)


        self.elbo = tf.reduce_mean( 
            self.log_likelihood_func(post_theta_samples, self.data_plc)
            + self.log_prior_func(post_theta_samples)
            - self.posterior_dist.log_prob(post_theta_samples) )

        self.learn_op = tf.train.AdamOptimizer().minimize(-self.elbo, var_list=list(self.posterior_dist.param.values()))


        self.full_data_posterior_dist = approximate_dist_class(self.nparam, **approximate_dist_config)

        self.log_threshold = tf.Variable(0.1, dtype=tf.float64)

        full_data_posterior_logprob_theta_samples = self.full_data_posterior_dist.log_prob(post_theta_samples)

        log_likelihood = tf.where(tf.math.greater(full_data_posterior_logprob_theta_samples, self.log_threshold),
            x = self.log_likelihood_func(post_theta_samples, self.data_plc),
            y = tf.zeros(nsample, dtype=tf.float64))

        self.eubo = tf.reduce_mean(
                    log_likelihood 
                    + self.posterior_dist.log_prob(post_theta_samples)
                    - full_data_posterior_logprob_theta_samples)
                    # - self.full_data_posterior_dist.log_prob(post_theta_samples))
        self.unlearn_op = tf.train.AdamOptimizer().minimize(self.eubo, var_list=list(self.posterior_dist.param.values()))


        full_post_theta_samples = self.full_data_posterior_dist.sample(nsample)
        # (nsample, nparam)

        full_data_post_logprob_full_theta_samples = self.full_data_posterior_dist.log_prob(full_post_theta_samples)

        log_likelihood_removed_data = self.log_likelihood_func(full_post_theta_samples, self.data_plc)
        # (nsample,)
        # log_likelihood_removed_data is also the weights

        # compute the average of negative log likelihood of removed data
        # where full_data_post_logprob_full_theta_samples >= log_threshold
        log_likelihood_removed_data = tf.where(
            tf.math.greater(full_data_post_logprob_full_theta_samples, self.log_threshold),
            x = log_likelihood_removed_data,
            y = tf.zeros_like(log_likelihood_removed_data, dtype=tf.float64))

        # log( E_q(x|full) 1/likelihood_of_removed_data )
        self.log_mean_inverse_likelihood = tf.reduce_logsumexp(
            -log_likelihood_removed_data - tf.cast(np.log(nsample), dtype=tf.float64))

        log_weights = -log_likelihood_removed_data - self.log_mean_inverse_likelihood

        self.unlearn_elbo = tf.reduce_mean(
            tf.exp(log_weights) * self.posterior_dist.log_prob(full_post_theta_samples))

        self.unlearn_elbo_op = tf.train.AdamOptimizer().minimize(- self.unlearn_elbo, 
                    var_list=list(self.posterior_dist.param.values()))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def learn(self, data, init_post_param = None, ntrain=1000, batchsize=1000000):
        """
        data: (n,d)
        learning by maximizing the ELBO
            = E_q(theta) [ logp(data|theta) ] - KL[q(theta) || p(theta)]
            = E_q(theta) [ logp(data|theta) - logq(theta) + logp(theta) ]
        """

        if init_post_param is not None:
            self.posterior_dist.load_param(init_post_param, self.sess)

        ndata = data.shape[0]
        nstart = 0
        idxs = np.array(list(range(ndata)))
        np.random.shuffle(idxs)
        
        start_time = time.time()

        for i in range(ntrain):
            self.sess.run( self.learn_op,
                feed_dict = {self.data_plc: 
                data[idxs[nstart:(nstart+batchsize)],...]} )
            nstart += batchsize

            if nstart >= ndata:
                nstart = 0
                np.random.shuffle(idxs)

            if i % 5000 == 0:
                print("{}. {:.4f} in {:.4f}s".format(i,
                    self.sess.run(self.elbo, feed_dict = {self.data_plc: data}),
                    time.time() - start_time))
                start_time = time.time()
                sys.stdout.flush()

        learned_param = {}
        for key in self.posterior_dist.param:
            learned_param[key] = self.sess.run(self.posterior_dist.param[key])

        return learned_param


    def unlearn_EUBO(self, data, full_data_post_param, log_threshold=0.1, ntrain=1000, batchsize=1000000):
        """
        data: (n,d)
        learning by minimizing the EUBO
            = E_q(theta) [ logp(data|theta) ] + KL[q(theta) || p(theta|alldata)]
            = E_q(theta) [ logp(data|theta) + logq(theta) - logp(theta|alldata) ]
        """

        print("Log threshold: {}".format(log_threshold))
        self.log_threshold.load(log_threshold, self.sess)

        self.posterior_dist.load_param(full_data_post_param, self.sess)
        self.full_data_posterior_dist.load_param(full_data_post_param, self.sess)
        
        ndata = data.shape[0]
        nstart = 0
        idxs = np.array(list(range(ndata)))
        np.random.shuffle(idxs)
        
        start_time = time.time()

        for i in range(ntrain):
            self.sess.run(self.unlearn_op, 
                feed_dict = {self.data_plc: 
                data[idxs[nstart:(nstart+batchsize)],...]} )
            nstart += batchsize

            if nstart >= ndata:
                nstart = 0
                np.random.shuffle(idxs)

            if i % 5000 == 0:
                print("{}. {:.4f} in {:.4f}s".format(i,
                    self.sess.run(self.eubo, feed_dict = {self.data_plc: data}),
                    time.time() - start_time))

                start_time = time.time()
                sys.stdout.flush()
                
        learned_param = {}
        for key in self.posterior_dist.param:
            learned_param[key] = self.sess.run(self.posterior_dist.param[key])

        return learned_param


    def unlearn_ELBO(self, data, full_data_post_param, log_threshold=0.0, log_scale=6.0, ntrain=1000, batchsize=1000000):
        """
        data: (n,d)
        learning by maximizing ELBO_unlearn
        """
        print("Log threshold: {}".format(log_threshold))
        self.log_threshold.load(log_threshold, self.sess)

        self.posterior_dist.load_param(full_data_post_param, self.sess)
        self.full_data_posterior_dist.load_param(full_data_post_param, self.sess)

        ndata = data.shape[0]
        nstart = 0
        idxs = np.array(list(range(ndata)))
        np.random.shuffle(idxs)

        start_time = time.time()

        for i in range(ntrain):
            self.sess.run(
                self.unlearn_elbo_op, 
                feed_dict = {
                    self.data_plc: 
                    data[idxs[nstart:(nstart+batchsize)],...]
                    } )
            nstart += batchsize

            if nstart >= ndata:
                nstart = 0
                np.random.shuffle(idxs)

            if i % 5000 == 0:
                print("{}. {:.4f} in {:.4f}s".format(i,
                    self.sess.run(self.unlearn_elbo, feed_dict = {
                        self.data_plc: data
                        } ),
                    time.time() - start_time))

                start_time = time.time()
                sys.stdout.flush()
                
        learned_param = {}
        for key in self.posterior_dist.param:
            learned_param[key] = self.sess.run(self.posterior_dist.param[key])

        return learned_param
