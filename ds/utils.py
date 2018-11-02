
def standard_scaler(x, mean, stdv):
  return (x-mean)/stdv

def maxmin_scaler(x, max_value, min_value):
  return (x-min_value)/(max_value-min_value)

