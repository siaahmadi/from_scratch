# Diff between `np.unstack` and `np.split`

`np.unstack` simply moves the given axis to the 0th position (`tuple(np.moveaxis(x, axis, 0))`)
`np.split` creates a new dimension (then chunks along the given axis and moves the chunks to the 0th axis)

```
a = np.random.standard_normal((5, 12))

np.array(np.unstack(a, axis=1)).shape  # --> (12, 5)
np.array(np.split(a, 3, axis=1)).shape # --> (3, 5, 4)
```

# Diff between `np.stack` and `np.concatenate`

* `np.stack`:  simply creates a new dimension (given by axis)
    and makes the result accessible as though it were a Python list. (“add an axis, then concatenate”).
    
    Does not allow different shapes.

* `np.concat`: allows different shapes along concatenation axis sums the sizes along concatenation axis

