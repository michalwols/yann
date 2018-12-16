def zero_out_nans(module, grad_input, grad_output):
  """
  Backwards hook to turn NaNs in gradients to 0
  """
  for grad in grad_input:
    grad[grad != grad] = 0  # technically shouldn't modify inputs
