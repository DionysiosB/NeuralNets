class Dual:

    def __init__(self, real:float = 0.0, dual:float = 0.0):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Dual(self.real + other, self.dual)
        elif isinstance(other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        else: raise NotImplementedError("Can only add Duals with other Duals or Numbers")

    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Dual(self.real - other, self.dual)
        elif isinstance(other, Dual):
            return Dual(self.real - other.real, self.dual - other.dual)
        else: raise NotImplementedError("Can only subtract Duals with other Duals or Numbers")

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Dual(self.real * other, self.dual * other)
        elif isinstance(other, Dual):
            return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)
        else: raise NotImplementedError("Can only multiply Duals with other Duals or Numbers")

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            if other == 0: raise ZeroDivisionError
            return Dual(self.real / other, self.dual / other)
        elif isinstance(other, Dual):
            return Dual(self.real / other.real, (other.real * self.dual - self.real * other.dual) / (other.real * other.real))
        else: raise NotImplementedError("Can only multiply Duals with other Duals or Numbers")

    def __pow__(self, other):
        if not isinstance(other, (float, int)):
            raise NotImplementedError("Can only raise Duals to a number (int or float) power")
        return Dual( (self.real)**p, other * ((self.real)**(p - 1)) * self.dual)
    
    def __neg__(self):
        return Dual(-self.real, -self.dual)

    __iadd__ = __add__
    __radd__ = __add__
    __isub__ = __sub__
    __rsub__ = __sub__
    __imul__ = __mul__
    __rmul__ = __mul__
    __itruediv__ = __truediv__
    __rtruediv__ = __truediv__

    def __str__(self):
        return f"Dual({self.real}, {self.dual})"

    def __bool__(self):
        return self.real > 0

    def __lt__(self, other):
        raise NotImplementedError("Comparison between Dual numbers is not defined")

    def __le__(self, other):
        raise NotImplementedError("Comparison between Dual numbers is not defined")

    def __gt__(self, other):
        raise NotImplementedError("Comparison between Dual numbers is not defined")

    def __ge__(self, other):
        raise NotImplementedError("Comparison between Dual numbers is not defined")

