import tensorflow as tf 
import numpy as np 




def chol2inv(mat, dtype=tf.float64):
    n = tf.shape(mat)[0]

    invlower = tf.matrix_solve(tf.linalg.cholesky(mat), 
                               tf.eye(n, dtype=dtype))
    invmat = tf.transpose(invlower) @ invlower
    return invmat


def computeKnm(X, Xbar, l, sigma, dtype=tf.float64):
    """
    X: n x d
    l: d
    """
    n = tf.shape(X)[0]
    m = tf.shape(Xbar)[0]

    X = X * tf.sqrt(l)
    Xbar = Xbar * tf.sqrt(l)

    Q = tf.tile(tf.reduce_sum( tf.square(X), axis=1 , keepdims=True ), multiples=(1,m))
    Qbar = tf.tile(tf.transpose(tf.reduce_sum(tf.square(Xbar), axis=1, keepdims=True )), multiples=(n,1)) 

    dist = Qbar + Q - 2 * X @ tf.transpose(Xbar)
    knm = sigma * tf.exp( -0.5 * dist )
    return knm


def computeKmm(X, l, sigma, dtype=tf.float64):
    """
    X: n x d
    l: 1 x d
    sigma: signal variance
    sigma * exp( - 0.5 * (X - X)^2 * lengthscale)
    X' = X * a
    lengthscale' = lengthscale / a^2
    """
    n = tf.shape(X)[0]
    X = X * tf.sqrt(l)
    Q = tf.tile(tf.reduce_sum( tf.square(X), axis=1, keepdims=True ), multiples=(1,n))
    dist = Q + tf.transpose(Q) - 2 * X @ tf.transpose(X)

    kmm = sigma * tf.exp(-0.5 * dist)
    return kmm



def compute_mean_f(x, Xsamples, Fsamples, l, sigma, 
                    KInv=None, dtype=tf.float64):
    """
    l: 1 x d
    Fsamples: k x m
    Xsamples: m x d
    x: n x d

    return: mean: k,n
    """
    if KInv is None:
        K = computeKmm(Xsamples, l, sigma, dtype=dtype)
        KInv = chol2inv( K, dtype=dtype )
        # m x m 

    kstar = computeKnm(x, Xsamples, l, sigma, dtype=dtype)
    # n x m

    mean = tf.expand_dims(kstar, axis=0) \
            @ (
                tf.expand_dims(KInv, axis=0) 
                @ tf.expand_dims(Fsamples, axis=2) 
            )
    # (k, n, 1)

    return tf.squeeze(mean, axis=2)
    # (k, n)


def compute_mean_var_f_marginalize_u(
                    x, Xsamples,
                    l, sigma, 
                    mean_fsamples,
                    cov_fsamples,
                    KInv=None, dtype=tf.float64):
    """
    l: 1 x d
    Xsamples: m x d
    mean_fsamples: (m,)
    cov_fsamples: (m,m)
    x: n x d
    KInv: m x m
    return: mean: k,n
    """
    m = tf.shape(Xsamples)[0]

    if KInv is None:
        K = computeKmm(Xsamples, l, sigma, dtype=dtype)
        KInv = chol2inv( K, dtype=dtype )
        # m x m 

    kstar = computeKnm(x, Xsamples, l, sigma, dtype=dtype)
    # n x m

    A = kstar @ KInv
    # n x m 

    var = sigma - tf.reduce_sum( A * kstar, axis=1 )
    # (n,)
    var = tf.clip_by_value(var, clip_value_min=0.0, clip_value_max=np.infty)
    
    var = var + tf.reduce_sum( (A @ cov_fsamples) * A, axis=1 )
    # (n,)

    mean = kstar @ (
                KInv 
                @ tf.reshape(mean_fsamples, shape=(m,1)) )
    # (n,1)
    
    return tf.squeeze(mean, axis=1), var
    # (n,) (n,)

