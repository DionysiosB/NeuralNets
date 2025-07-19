from dualnumbers import Dual

def derivative(f = None, x: float = 0.0):
  xeps = Dual(x, 1)
  diff = f(xeps) - f(x)
  return diff.dual


def f(x: float = 0.0):
  return x**3 + 2 * x**2 + 3 * x + 5

print(derivative(f, 0))
