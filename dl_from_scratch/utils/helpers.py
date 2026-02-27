import numpy as np

def gradient_check(func, x, func_output_shape=(), h=1e-5, **kw_args):
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

def softmax(x, axis=None, compute_grad=False, upstream_grad=None):
    max_vals = np.max(x, axis=axis, keepdims=True)
    max_vals[~np.isfinite(max_vals)] = 0

    x = x - max_vals
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    sum_exp_x[sum_exp_x==0] = 1 # avoid div-by-zero
    p = exp_x / sum_exp_x # (b, h, L_q, L_k)

    grad = None
    if compute_grad:
        if upstream_grad is None: # form jacobian
            pT = np.expand_dims(p, len(p.shape))
            p_expanded = np.expand_dims(p, -2)
            p_diag = p_expanded * np.eye(p.shape[-1])
            grad = p_diag - pT @ p_expanded
        else: # no jacobian needed; upstream_gradient of shape (b, h, L_q, L_k)
            g_dot_p = np.vecdot(upstream_grad, p)[..., np.newaxis]  # (b, h, L_q, 1)
            grad = (upstream_grad - g_dot_p) * p # (b, h, L_q, L_k) * [(b, h, L_q, L_k) - (b, h, L_q, 1)]

    return (p, grad) if compute_grad else p

def masked_softmax(x, mask, axis=None, upstream_grad=None):
    y = x.copy()
    if mask is not None:
        mask = mask.astype(bool)
        y[mask] = -np.inf
    return softmax(y, axis=axis, upstream_grad=upstream_grad)

def cross_entropy_loss(logits, p):
    logits -= np.max(logits, axis=1, keepdims=True)

    correct_class = np.argmax(p, axis=1)
    row = np.arange(logits.shape[0])
    return np.log(np.exp(logits).sum(axis=1)) - logits[row, correct_class]

def cross_entropy_gradient(q, p):
    return q - p

def loss_and_accuracy(model, X, true_label):

    logits = model(X)

    loss = cross_entropy_loss(logits, true_label).mean()

    predicted = np.argmax(softmax(logits), axis=1)
    true = np.argmax(true_label, axis=1)
    accuracy = np.mean(predicted == true)
    
    return loss, accuracy

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

    G = np.ones_like(x)
    smx, smx_grad = softmax(x, axis=-1, compute_grad=True, upstream_grad=None)
    numeric_grad = gradient_check(softmax, x, func_output_shape=x.shape[-1:], axis=-1)
    print(f"Gradient check pass? {np.allclose(numeric_grad, smx_grad, rtol=1e-8)}")
    print()

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