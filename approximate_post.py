import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp 



class Gaussian():

    def __init__(self, dim):
        self.dim = dim
        
        self.param = {'loc': tf.Variable(tf.zeros(self.dim, dtype=tf.float64), dtype=tf.float64),
                    'sqrt_cov': tf.Variable(tf.eye(self.dim, dtype=tf.float64), dtype=tf.float64)}
        sqrt_cov = tf.linalg.band_part(self.param['sqrt_cov'], -1, 0)

        self.cov = sqrt_cov @ tf.linalg.transpose(sqrt_cov)

        self.dist = tfp.distributions.MultivariateNormalFullCovariance(
                            loc = self.param['loc'],
                            covariance_matrix = self.cov)


    def sample(self, size):
        samples = self.dist.sample(size)
        samples = tf.reshape(samples, shape=(size,self.dim))
        return samples


    def log_prob(self, x):
        # (:,self.dim)
        logprobs = self.dist.log_prob(x)
        return tf.reshape(logprobs, shape=tf.shape(x)[:-1])

    def load_param(self, new_param, session):
        for key in self.param:
            self.param[key].load(new_param[key], session)

    def find_mode(self, param=None, n_init=10, ntrain=100):
        mode_log_prob = self.log_prob(tf.reshape(self.param['loc'], shape=(1,self.dim)))

        with tf.Session() as sess:
            if param is None:
                sess.run(tf.global_variables_initializer())
            else:
                sess.run(tf.global_variables_initializer())
                self.load_param(param, sess)

            mode_x_np, mode_log_prob_np = sess.run([self.param['loc'], mode_log_prob])

        return mode_x_np.reshape(1,self.dim), mode_log_prob_np.squeeze()


class GaussianDiag():

    def __init__(self, dim):
        self.dim = dim
        
        self.param = {'loc': tf.Variable(tf.zeros(self.dim, dtype=tf.float64), dtype=tf.float64),
                    'log_scale': tf.Variable(tf.ones(self.dim, dtype=tf.float64), dtype=tf.float64)}
        scale = tf.exp(self.param['log_scale'])

        self.dist = tfp.distributions.MultivariateNormalDiag(
                            loc = self.param['loc'],
                            scale_diag = scale)

        
    def sample(self, size):
        samples = self.dist.sample(size)
        samples = tf.reshape(samples, shape=(size,self.dim))
        return samples


    def log_prob(self, x):
        # (:,self.dim)
        logprobs = self.dist.log_prob(x)
        return tf.reshape(logprobs, shape=tf.shape(x)[:-1])

    def load_param(self, new_param, session):
        for key in self.param:
            self.param[key].load(new_param[key], session)

    def find_mode(self, param=None, n_init=10, ntrain=100):
        mode_log_prob = self.log_prob(tf.reshape(self.param['loc'], shape=(1,self.dim)))

        with tf.Session() as sess:
            if param is None:
                sess.run(tf.global_variables_initializer())
            else:
                sess.run(tf.global_variables_initializer())
                self.load_param(param, sess)

            mode_x_np, mode_log_prob_np = sess.run([self.param['loc'], mode_log_prob])

        return mode_x_np.reshape(1,self.dim), mode_log_prob_np.squeeze()


class MAF():
    class_count = 0

    def __init__(self, dim, 
                nbijectors=5, 
                hidden_layers=[5,5]):
        
        self.dim = dim
        self.name = "MAF_{}".format(MAF.class_count)

        import maf
        self.maf = maf.MAF(dim, nbijectors, hidden_layers, name=self.name)
        MAF.class_count += 1

        self.maf.sample(1) # dummy, so that tf.trainable_variables() does not return empty list

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        trainable_names = ["var_{}".format(i) for i in range(len(trainable_vars))]
        self.param = dict(zip(trainable_names, trainable_vars))


    def sample(self, size):
        return self.maf.sample(size)


    def log_prob(self, x):
        # x: [:,self.dim]
        logprobs = self.maf.log_prob(x)
        return tf.reshape(logprobs, shape=tf.shape(x)[:-1])


    def load_param(self, new_param, session):
        for key in self.param:
            self.param[key].load(new_param[key], session)


    def find_mode(self, param=None, n_init=10, ntrain=100):
        
        samples = self.sample(n_init)

        initializers = tf.Variable(tf.zeros_like(samples), dtype=tf.float64, name="initializers")
        # (n_init, self.dim)

        assign = tf.assign(initializers, samples)

        logprobs = self.log_prob(initializers)
        # (n_init,)

        loss = - tf.reduce_sum(logprobs)

        train = tf.train.AdamOptimizer().minimize(loss, var_list=[initializers])

        with tf.Session() as sess:
            if param is None:
                sess.run(tf.global_variables_initializer())
            else:
                sess.run(tf.global_variables_initializer())
                self.load_param(param, sess)

            sess.run(assign)

            for i in range(ntrain):
                sess.run(train)

            init_np, lprob_np = sess.run([initializers, logprobs])

        max_idx = np.argmax(lprob_np)
        return init_np[max_idx,:].reshape(1,self.dim), lprob_np[max_idx]


def minimize_kl(dim, 
        opt_dist_class, opt_dist_config, 
        ref_dist_class, ref_dist_config, 
        ref_dist_param, 
        nsample = 100,
        ntrain = 1000):

    opt_dist = opt_dist_class(dim, **opt_dist_config)
    ref_dist = ref_dist_class(dim, **ref_dist_config)

    opt_samples = opt_dist.sample(nsample)
    kl = tf.reduce_mean(opt_dist.log_prob(opt_samples) - ref_dist.log_prob(opt_samples))

    train = tf.train.AdamOptimizer().minimize(kl)
    
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        ref_dist.load_param(ref_dist_param, sess)

        for i in range(ntrain):
            sess.run(train)

        opt_param = {}
        for key in opt_dist.param:
            opt_param[key] = sess.run(opt_dist.param[key])
        
    return opt_param





