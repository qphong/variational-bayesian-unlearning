import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np 



class MAF():

    def __init__(self, dim, 
                nbijectors=5, 
                hidden_layers=[512,512],
                name = "MAF"):

        self.dim = dim 

        # create masked autoregressive flow (MAF)
        self.base_dist = tfp.distributions.MultivariateNormalDiag(
            loc = tf.zeros(self.dim, dtype=tf.float64))

        self.nbijectors = nbijectors
        self.hidden_layers = hidden_layers

        self.bijectors = []

        for _ in range(self.nbijectors):
            self.bijectors.append(
                tfp.bijectors.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn = tfp.bijectors.masked_autoregressive_default_template(
                        hidden_layers = self.hidden_layers,
                        log_scale_clip_gradient=True,
                        name=name)
                )
            )

            self.bijectors.append(
                tfp.bijectors.Permute(
                    permutation=list(reversed(range(self.dim)))
                )
            )

        # discard the last Permute layer
        flow_bijector = tfp.bijectors.Chain(
                    list(reversed(self.bijectors[:-1])))
        self.dist = tfp.distributions.TransformedDistribution(
                distribution=self.base_dist,
                bijector=flow_bijector,
                name='maf_dist')


    def sample(self, size):
        all_layers_fw_samples = [None] * (self.nbijectors * 2)
        all_layers_fw_samples[0] = self.base_dist.sample(size)
        self.layers_names = [self.base_dist.name]

        for i,bijector in enumerate(reversed(self.dist.bijector.bijectors)):
            all_layers_fw_samples[i+1] = bijector.forward(all_layers_fw_samples[i])
            self.layers_names.append(bijector.name)

        return all_layers_fw_samples[-1]


    def log_prob(self, X):
        # X: (...,dim)
        return self.dist.log_prob(X)
        # (...)


