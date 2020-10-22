import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp 
import scipy.stats
import sklearn.datasets as skdata
import gp_utils 


class GammaUnknownShape():
    def __init__(self, dim, nparam, params, prior_params):
        """
        params: dictionary {"log_shape", "log_rate"} of numpy arrays
        prior_params:
            dict: "weight" np.array of size k
                  "dist" list of dicts of size k
                        dict: "loc"
                              "cov"
        """
        self.dim = dim 
        self.nparam = nparam

        self.params = params # ground truth
        self.prior_params = prior_params # loc, cov

        
        prior_mixture_size = prior_params['weight'].shape[0]

        self.log_prior_weight = tf.cast(tf.log(prior_params['weight'].reshape(-1,1)), dtype=tf.float64)
        # (len(self.prior), 1)

        self.prior_cumsum_weights = np.cumsum(prior_params['weight'])
        self.prior = []
        
        for i in range(prior_mixture_size):
            self.prior.append(
                tfp.distributions.MultivariateNormalFullCovariance(
                        loc = self.prior_params["dist"][i]["loc"], 
                        covariance_matrix = self.prior_params["dist"][i]["cov"])
            )

    def log_prior(self, theta):
        """
        theta: (nsample, nparam)
        return (nsample,)
        """
        nsample = tf.shape(theta)[0]
        probs = []

        for i in range(len(self.prior)):
            probs.append( self.prior[i].log_prob(theta) )

        prior_probs = tf.stack(probs) + self.log_prior_weight
        # (len(self.prior), nsample)
        
        return tf.reduce_logsumexp(prior_probs, axis=0)
        # (nsample,)


    def log_likelihood(self, theta, data, sum_over_data=True):
        """
        theta: (nsample,nparam)
        data: (ndata,d)
        return (nsample,)
        """
        nsample = tf.shape(theta)[0]

        likelihood = tfp.distributions.Gamma(
                        concentration = tf.exp(tf.expand_dims(theta, axis=0)), 
                        rate = tf.expand_dims(tf.tile(tf.expand_dims(tf.exp(self.params['log_rate']), axis=0), multiples=(nsample,1)), axis=0))

        log_probs = tf.squeeze(likelihood.log_prob(
                tf.tile(tf.expand_dims(data, axis=1), multiples=(1,nsample,1))
            ), axis=-1)
        # (ndata, nsample)

        if sum_over_data:
            return tf.reduce_sum(log_probs, axis=0)
            # (nsample,)
        else:
            return log_probs
            # (ndata, nsample)


    def sample_data_np(self, size, xbound=None, seed=0):
        """
        xbound is dummy (not used)
        """
        np.random.seed(seed)
        print("True param:", self.params)

        data = scipy.stats.gamma.rvs(
                a=np.exp(self.params["log_shape"]), 
                scale=1.0/np.exp(self.params['log_rate']),
                size=size)
        data = data.reshape(size,self.dim)

        return data


class UCILogisticRegression():
    def __init__(self, dataset, order, prior_gaussian_mean = 0, prior_gaussian_std = 10000):
        """
        data dimension:
            dim = dim_x + dim_y
            dim_y = 1
        
        parameter dimension: given k-classification problem
            if order = 1:
                nparam = dim * (k-1) = (dim_x + 1) * (k-1) 
                (dim_x + 1: weights and bias)
            elif order = 2:
                nparam = (dim_x * (dim_x-1) / 2 + 2 * dim_x + 1) * (k-1)
                for example dim_x=2 and k=2: 
                    x = [x0,x1]
                    model: f(x) = w0*x0 + w1*x1 + w2*x0^2 + w3*x1^2 + w4*x0*x1 + bias 
        """
        import pandas as pd 

        self.dataset = dataset 

        if self.dataset == "banknote":
            self.path = "dataset/banknote_authen/data_banknote_authentication.txt"
            self.data = pd.read_csv(self.path, index_col=False, 
                        names = ["var", "skew", "curt", "entropy", "class"])
        else:
            raise Exception("Unknown dataset: {}".format(self.dataset))

        self.dim = self.data.shape[1]
        self.nclass = len(self.data["class"].unique())
        self.order = order 

        if self.order == 1:
            self.nparam_per_class = self.dim
            self.nparam = int(self.nparam_per_class * (self.nclass - 1))
        
        elif self.order == 2:
            self.nparam_per_class = int((self.dim-1) * (self.dim-2) / 2 + 2 * (self.dim-1) + 1)
            self.nparam = int(self.nparam_per_class * (self.nclass - 1))

        else:
            raise Exception("Haven't implemented for order > 2")

        # prior distribution is Gaussian
        self.prior = tfp.distributions.MultivariateNormalDiag(
                        loc = tf.zeros(self.nparam, dtype=tf.float64) + prior_gaussian_mean, 
                        scale_diag = tf.ones(self.nparam, dtype=tf.float64) * prior_gaussian_std)


    def log_prior(self, theta):
        """
        theta: (nsample, nparam)
        return (nsample,)
        """
        return self.prior.log_prob(theta)


    def log_likelihood(self, theta, data, sum_over_data=True):
        """
        theta: (nsample,nparam)
        data: (ndata,dim)
        return (nsample,)
        """
        nsample = tf.shape(theta)[0]
        ndata = tf.shape(data)[0]

        x = tf.gather(data, indices=list(range(self.dim-1)), axis=1)
        # (ndata,dim-1)
        y = tf.cast(tf.gather(data, indices=self.dim-1, axis=1), dtype=tf.int32)
        # (ndata,)

        if self.order == 1:
            x = tf.concat([x, tf.ones((ndata,1), dtype=tf.float64)], axis=1)
            # (ndata,dim)

        elif self.order == 2:
            x2 = x * x

            inter_x = []
            for i in range(self.dim-1):
                xi = tf.gather(x, indices=[i], axis=1)

                for j in range(i+1,self.dim-1):
                    xj = tf.gather(x, indices=[j], axis=1)
                    inter_x.append( xi * xj )
            
            inter_x = tf.concat(inter_x, axis=1)

            x = tf.concat([x, x2, inter_x, tf.ones((ndata,1), dtype=tf.float64)], axis=1)
            # (ndata, self.nparam_per_class)

        else:
            raise Exception("Haven't implemented for order > 2")


        x = tf.expand_dims(x, axis=1)
        # (ndata, 1, self.nparam_per_class)
        
        fx = []
        for i in range(self.nclass-1):
            theta_i = tf.gather(theta, 
                        indices=list(range( i*self.nparam_per_class, 
                                            (i+1)*self.nparam_per_class )), 
                        axis=1)
            # (nsample, nparam_per_class)

            fx.append( tf.reduce_sum(x * theta_i, axis=-1) )
            # (ndata, nsample)
        
        # the lass class has probability 1 / (1 + sum_{i=1}^{nclass-1} exp(f_i(x)))
        fx.append(tf.zeros_like(fx[-1], dtype=tf.float64))

        fx = tf.stack(fx)
        # (self.nclass, ndata, nsample)
        
        indices = tf.stack( [y, tf.range(ndata, dtype=tf.int32)] )
        # (2, ndata)
        indices = tf.transpose(indices)
        # (ndata, 2)

        selected_fx = tf.gather_nd(fx, indices)
        # (ndata, nsample)

        logprobs = selected_fx - tf.reduce_logsumexp(fx, axis=0)
        # (ndata, nsample)

        if sum_over_data:
            return tf.reduce_sum(logprobs, axis=0)
            # (nsample,)
        else:
            return logprobs
            # (ndata, nsample)


    def log_posterior_predictive(self, theta, data):
        """
        theta: (nsample,nparam)
        # only use the input in data
        # compute the predictive prob over all possible outputs
        data: (ndata,dim)
        return (nsample,)
        """
        nsample = tf.shape(theta)[0]
        ndata = tf.shape(data)[0]

        x = tf.gather(data, indices=list(range(self.dim-1)), axis=1)
        # (ndata,dim-1)
        y = tf.cast(tf.gather(data, indices=self.dim-1, axis=1), dtype=tf.int32)
        # (ndata,)

        if self.order == 1:
            x = tf.concat([x, tf.ones((ndata,1), dtype=tf.float64)], axis=1)
            # (ndata,dim)

        elif self.order == 2:
            x2 = x * x

            inter_x = []
            for i in range(self.dim-1):
                xi = tf.gather(x, indices=[i], axis=1)

                for j in range(i+1,self.dim-1):
                    xj = tf.gather(x, indices=[j], axis=1)
                    inter_x.append( xi * xj )
            
            inter_x = tf.concat(inter_x, axis=1)

            x = tf.concat([x, x2, inter_x, tf.ones((ndata,1), dtype=tf.float64)], axis=1)
            # (ndata, self.nparam_per_class)

        else:
            raise Exception("Haven't implemented for order > 2")


        x = tf.expand_dims(x, axis=1)
        # (ndata, 1, self.nparam_per_class)
        
        fx = []
        for i in range(self.nclass-1):
            theta_i = tf.gather(theta, 
                        indices=list(range( i*self.nparam_per_class, 
                                            (i+1)*self.nparam_per_class )), 
                        axis=1)
            # (nsample, nparam_per_class)

            fx.append( tf.reduce_sum(x * theta_i, axis=-1) )
            # (ndata, nsample)
        
        # the lass class has probability 1 / (1 + sum_{i=1}^{nclass-1} exp(f_i(x)))
        fx.append(tf.zeros_like(fx[-1], dtype=tf.float64))

        fx = tf.stack(fx)
        # (self.nclass, ndata, nsample)
        
        logprobs = fx - tf.reduce_logsumexp(fx, axis=0, keepdims=True)
        # (self.nclass, ndata, nsample)

        return logprobs
        # (nclass, ndata, nsample)


    def sample_data_np(self, size, xbound=None, seed=0):
        """
        dim = dim_x + dim_y
        dim_y = 1

        dim_x = number of order of the function
            e.g., dim_x = 1: ax + b
                dim_x = 2: ax^2 + bx + c
                dim_x = 3: ax^3 + bx^2 + cx + d
        """
        print("sample_data_np: Reading from dataset: {}. Ignore xbound".format(self.dataset))
        
        np.random.seed(seed)
        data_np = self.data.values

        idxs = np.array(list(range(data_np.shape[0])))
        np.random.shuffle(idxs)

        return data_np[idxs[:size], :]



class MoonClassification():
    """
    moon classification dataset from sklearn
    parameters: f(x) where x is in a set of inducing inputs
    likelihood: using GP posterior given f(x)
        then logit function: 
            p(c_x = 1) = exp(f(x)) / (exp(f(x)) + 1)
            p(c_x = 0) = 1         / (exp(f(x)) + 1)
    """
    def __init__(self, nu=20, noise_std=0.2,
            log_lengthscales=np.log(np.array([2.4232, 1.8230])),
            log_sigma=np.log(4.7432)):
        self.nu = nu
        self.noise_std = noise_std
        self.xu, _ = skdata.make_moons(n_samples=nu, shuffle=True, 
                                noise=noise_std, random_state=0)
        # xu: (nu,2)
        
        self.dim = 3 # first 2 dim are data, last dim is label
        self.nparam = self.nu # inducing variables at inducing inputs

        if log_lengthscales is None or log_sigma is None:
            print("lengthscales and sigma are trainable")    
            self.log_lengthscales = tf.Variable(np.zeros(self.dim-1), dtype=tf.float64)
            self.log_sigma = tf.Variable(0.0, dtype=tf.float64)
        else:
            self.log_lengthscales = tf.constant(log_lengthscales, dtype=tf.float64)
            self.log_sigma = tf.constant(log_sigma, dtype=tf.float64)

        self.lengthscales = tf.exp(self.log_lengthscales) 
        self.sigma = tf.exp(self.log_sigma)
        
        self.K = gp_utils.computeKmm(
                    self.xu, 
                    self.lengthscales, 
                    self.sigma, 
                    dtype=tf.float64)
        self.KInv = gp_utils.chol2inv(self.K)

        self.prior = tfp.distributions.MultivariateNormalFullCovariance(
            loc = tf.zeros(self.nu, dtype=tf.float64),
            covariance_matrix = self.K
        )


    def log_prior(self, theta):
        """
        theta: (nsample, nparam)
        return (nsample,)
        """
        return self.prior.log_prob(theta)
    

    def log_likelihood(self, theta, data, sum_over_data=True):
        """
        theta: (nsample,nparam)
        data: (ndata,dim)
        return (nsample,)
        """
        nsample = tf.shape(theta)[0]
        ndata = tf.shape(data)[0]

        x = tf.gather(data, indices=list(range(self.dim-1)), axis=1)
        # (ndata,dim-1)
        y = tf.cast(tf.gather(data, indices=self.dim-1, axis=1), dtype=tf.float64)
        # (ndata,)

        predicted_f = gp_utils.compute_mean_f(
            x,
            Xsamples = self.xu,
            Fsamples = theta,
            l = self.lengthscales,
            sigma = self.sigma,
            KInv = self.KInv,
            dtype = tf.float64
        )
        # (nsample,ndata)

        """
        exp(fx) / (1 + exp(fx))
        exp(fx/2) / (exp(-fx/2) + exp(fx/2))
        exp(y * fx - fx/2) / (exp(-fx/2) + exp(fx/2))
        """
        half_pred_f = predicted_f / 2.0
        normalizer = tf.stack([half_pred_f, - half_pred_f])
        # (2,nsample,ndata)

        logprobs = predicted_f * y - half_pred_f - tf.reduce_logsumexp(normalizer, axis=0)
        # (nsample,ndata)

        if sum_over_data:
            return tf.reduce_sum(logprobs, axis=1)
            # (nsample)
        else:
            return tf.transpose(logprobs)
            # (ndata,nsample)


    def log_posterior_predictive(self, theta, data):
        """
        theta: (nsample,nparam)
        # only use the input in data
        # compute predictive prob over all possible outputs
        data: (ndata,dim)
        return (2, ndata, nsample)
        """
        nsample = tf.shape(theta)[0]
        ndata = tf.shape(data)[0]

        x = tf.gather(data, indices=list(range(self.dim-1)), axis=1)
        # (ndata,dim-1)
        y = tf.cast(tf.gather(data, indices=self.dim-1, axis=1), dtype=tf.float64)
        # (ndata,)

        predicted_f = gp_utils.compute_mean_f(
            x,
            Xsamples = self.xu,
            Fsamples = theta,
            l = self.lengthscales,
            sigma = self.sigma,
            KInv = self.KInv,
            dtype = tf.float64
        )
        # (nsample,ndata)

        """
        exp(fx) / (1 + exp(fx))
        exp(fx/2) / (exp(-fx/2) + exp(fx/2))
        exp(y * fx - fx/2) / (exp(-fx/2) + exp(fx/2))
        # y can only be either 0 or 1
        """
        half_pred_f = predicted_f / 2.0
        normalizer = tf.stack([half_pred_f, - half_pred_f])
        normalizer = tf.reduce_logsumexp(normalizer, axis=0)
        # (nsample,ndata)

        logprobs1 = half_pred_f - normalizer
        logprobs2 = -half_pred_f - normalizer
        # (nsample, ndata)

        logprobs = tf.stack([tf.transpose(logprobs1), tf.transpose(logprobs2)])
        # (2, ndata, nsample)
        
        return logprobs


    def sample_data_np(self, size, noise_std=0.2, xbound=None, seed=0):
        X, labels = skdata.make_moons(n_samples=size, shuffle=True, noise=noise_std, random_state=seed)
        # X.shape = (n_samples,2)
        # labels.shape = (n_samples,)
        data = np.concatenate([X, labels.reshape(size,1)], axis=1)
        """
        data[:,0] -1.5, 2.5
        data[:,1] -1, 1.5
        """
        return data 


    def predict_f(self, x, mean_u, cov_u):
        # x: (nx,dim)
        # theta: (nparam,)

        meanf, varf = gp_utils.compute_mean_var_f_marginalize_u(
                    x, 
                    self.xu,
                    self.lengthscales, 
                    self.sigma, 
                    mean_u,
                    cov_u,
                    self.KInv, 
                    dtype=tf.float64)

        with tf.Session() as sess:
            meanf_np, varf_np = sess.run([meanf, varf])

        return meanf_np, varf_np

