import numpy as np

def gradient_check(func, x, func_output_shape=(), h=1e-5, **kw_args):
    """Numerical validation of analytic gradient implementation
    
    Iterates over each value of x and computes the central difference estimate:

                          f(x+h) - f(x-h)
    f'(x) =  lim_{h-->0} -----------------
                                2h
    
    This requires two forward passes but has error on the order of O(h^2),
    which is more desirable than O(h) error of the single-forward pass formula.

    args:
        func:
            function handle to compute the forwrad pass
        
        x:
            point at which to estimate the gradient
        
        func_output_shape:
            tuple to indicate `func`'s output shape; can be empty tuple () if `func` returns a scalar
            Note: if `func` outputs a vector, the gradient check will form the full Jacobian.
        
        h:
            step size (default: 1d-5).
        
        kw_args:
            keyword arguments to be passed to `func` to control its behavior.
            This function expects `func` to return a single argument (can be ndarray) so if the default
            behavior of `func` is to return more than one argument use kw_args to change that.
    """

    x = np.atleast_2d(x)
    it = np.nditer(x, ['multi_index'], ['readwrite'])
    grad = np.zeros(x.shape + func_output_shape)
    with it:
        while not it.finished:
            ix = it.multi_index
            prev_x = x[ix]
            x[ix] = prev_x + h
            fxh_r = func(x, **kw_args) # f(x + h)
            x[ix] = prev_x - h
            fxh_l = func(x, **kw_args) # f(x - h)
            x[ix] = prev_x

            ix_func_output = ix[:len(func_output_shape)]
            grad[ix] = ((fxh_r - fxh_l) / (2 * h))[ix_func_output]

            it.iternext() # step to next dimension
    
    return grad

def softmax(x, axis=None, return_grad=False, return_jacobian=False):
    """
    Numerically stable implementation of the softmax function

    Won't return NaN's if all entries are -infinity. Instead, these will be 0's.

    args:
        x          : tensor of inputs
        axis       : axis along which to compute the softmax
        return_grad: if True, the gradient function will be returned as a second argument

    NOTE: Currently the gradient is valid only if axis=-1!"""

    max_vals = np.max(x, axis=axis, keepdims=True)
    max_vals[~np.isfinite(max_vals)] = 0 # avoids NaNs if entire row is -infinity (don't want: -infinity - (-infinity) = undefined)

    x = x - max_vals
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    sum_exp_x[sum_exp_x==0] = 1 # avoid div-by-zero
    p = exp_x / sum_exp_x # (b, h, L_q, L_k)

    def jacobian():
        """Forms Jacobian of vector-valued softmax: R^n --> R^n (where n == x.shape[-1])"""

        if axis != -1 and axis != len(x) - 1:
            raise Exception("Currently, the gradient can be computed only when axis=-1.")

        pT = np.expand_dims(p, len(p.shape))
        p_expanded = np.expand_dims(p, -2)
        p_diag = p_expanded * np.eye(p.shape[-1])
        grad = p_diag - pT @ p_expanded

        return grad
    
    def grad(upstream_grad):
        """Computes the upstream loss function's gradient with respect to inputs x
        without forming the Jacobian first
        
        upstream_gradient must be the same shape as softmax's input (b, h, L_q, L_k)"""

        if axis != -1 and axis != len(x) - 1:
            raise Exception("Currently, the gradient can be computed only when axis=-1.")

        g_dot_p = np.vecdot(upstream_grad, p)[..., np.newaxis]  # (b, h, L_q, 1)
        grad = (upstream_grad - g_dot_p) * p # (b, h, L_q, L_k) * [(b, h, L_q, L_k) - (b, h, L_q, 1)]

        return grad
    
    grad_fn = jacobian if return_jacobian else grad

    return (p, grad_fn) if return_grad else p

def masked_softmax(x, mask, axis=None, return_grad=False):
    """
    Apply softmax function after masked out some elements.

    If an entire row is masked out, that row will contain all 0's in the output (no NaNs).

    args:
        x          : tensor of inputs
        mask       : boolean mask. True entries will be masked. May be non-boolean if it can be converted to valid boolean.
        axis       : axis along which to compute the softmax
        return_grad: if True, the gradient function will be returned as a second argument
    """

    y = x.copy()
    if mask is not None:
        mask = mask.astype(bool)
        y[mask] = -np.inf

    return softmax(y, axis=axis, return_grad=return_grad)

def cross_entropy_loss(logits, p, return_grad=False):
    """
    args:
        logits     : unnormalized model outputs
        p          : one-hot encoded true class probability distributions
        return_grad: if True, the gradient function will be returned as a second argument
    """

    logits = logits - np.max(logits, axis=1, keepdims=True)
    
    correct_class = np.argmax(p, axis=-1)
    row = np.arange(logits.shape[0])
    loss = np.log(np.exp(logits).sum(axis=-1)) - logits[row, correct_class]

    def grad(upstread_grad):
        """Gradient of the cross entropy loss with respect to unnormalized model output
        
        args:
            upstread_grad: ignored. Assumed to be 1, and provided for API consistency only.
        """
        
        q = softmax(logits, axis=-1)
        return q - p

    return (loss, grad) if return_grad else loss

def cross_entropy_gradient(q, p):
    """Gradient of the cross entropy loss with respect to unnormalized model output
    args:
        q: estimated probabilities
        p: true probabilities
    """
    return q - p

def loss_and_accuracy(model, X, true_label):

    logits = model(X)

    loss = cross_entropy_loss(logits, true_label).mean()

    predicted = np.argmax(softmax(logits), axis=1)
    true = np.argmax(true_label, axis=1)
    accuracy = np.mean(predicted == true)
    
    return loss, accuracy

def accuracy(logits, true_label_onehot):

    predicted = np.argmax(softmax(logits), axis=-1)
    true = np.argmax(true_label_onehot, axis=-1)
    accuracy = np.mean(predicted == true, axis=-1)
    
    return accuracy

def xavier_uniform(shape, seed=None):
    rng = np.random.default_rng(seed=seed)
    d_in = shape[-2]
    var_factor = 1/np.sqrt(d_in)
    return rng.uniform(-1, 1, size=shape) * var_factor

def xavier_normal(shape, seed=None):
    rng = np.random.default_rng(seed=seed)
    d_in = shape[-2]
    var_factor = 1/np.sqrt(d_in)
    return rng.standard_normal(shape) * var_factor


if __name__ == "__main__":
    seed = 0

    rng = np.random.default_rng(seed=seed)

    N, D = 8, 4

    x = rng.standard_normal((N, D))

    mask = np.ones(x.shape, dtype=np.bool)

    y = masked_softmax(x, mask=mask, axis=1)

    print(y)

    G = np.ones_like(x) # or None
    
    smx, smx_grad = softmax(x, axis=-1, compute_grad=True, upstream_grad=G)
    numeric_grad = gradient_check(softmax, x, func_output_shape=x.shape[-1:], axis=-1)
    numeric_grad = numeric_grad if G is None else np.squeeze(np.expand_dims(G, -2) @ numeric_grad)
    print(f"Gradient check pass? {np.allclose(numeric_grad, smx_grad, rtol=1e-8)}")
    print()

    x = x[:min(N, D)] # make square
    causal_mask = np.triu(np.ones(x.shape, dtype=np.bool), 1)
    G = None # or np.ones_like(x)
    
    smx, smx_grad = masked_softmax(x, mask=causal_mask, axis=-1, compute_grad=True, upstream_grad=G)
    numeric_grad = gradient_check(masked_softmax, x, func_output_shape=x.shape[-1:], mask=causal_mask, axis=-1)
    numeric_grad = numeric_grad if G is None else np.squeeze(np.expand_dims(G, -2) @ numeric_grad)
    print(f"Gradient check pass? {np.allclose(numeric_grad, smx_grad, rtol=1e-8)}")
    print()

# with G = None
# p (softmax(x[:2]))
# array([[0.22597686, 0.17461744, 0.37808727, 0.22131843],
#        [0.07065947, 0.17331947, 0.44476298, 0.31125809]])
# 
# smx_grad of x[0]
# array([[[ 0.17491132, -0.0394595 , -0.08543897, -0.05001284],
#         [-0.0394595 ,  0.14412619, -0.06602063, -0.03864606],
#         [-0.08543897, -0.06602063,  0.23513729, -0.08367768],
#         [-0.05001284, -0.03864606, -0.08367768,  0.17233658]]])
# 
# smx_grad of x[1]
# array([[[ 0.06566671, -0.01224666, -0.03142671, -0.02199333],
#         [-0.01224666,  0.14327983, -0.07708608, -0.05394709],
#         [-0.03142671, -0.07708608,  0.24694887, -0.13843607],
#         [-0.02199333, -0.05394709, -0.13843607,  0.21437649]]])
#
# numeric_grad of x[:2]
# array([[[ 0.17491132, -0.0394595 , -0.08543897, -0.05001284],
#         [-0.0394595 ,  0.14412619, -0.06602063, -0.03864606],
#         [-0.08543897, -0.06602063,  0.23513729, -0.08367768],
#         [-0.05001284, -0.03864606, -0.08367768,  0.17233658]],

#        [[ 0.06566671, -0.01224666, -0.03142671, -0.02199333],
#         [-0.01224666,  0.14327983, -0.07708608, -0.05394709],
#         [-0.03142671, -0.07708608,  0.24694887, -0.13843607],
#         [-0.02199333, -0.05394709, -0.13843607,  0.21437649]],