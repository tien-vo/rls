__all__ = ["SimQuantity"]

import re

import numpy as np
from astropy.units import Quantity, StructuredUnit, isclose


class SimQuantity(Quantity):
    _dtype = np.dtype([("_code", "f8"), ("_user", "f8")])

    def __new__(cls, code_value, user_value=None, unit=""):
        if isinstance(user_value, Quantity):
            unit = str(user_value.unit)
            user_value = user_value.value
        elif user_value is None:
            user_value = code_value

        unit = StructuredUnit(("", unit))
        code_value = np.array(code_value, ndmin=0)
        user_value = np.array(user_value, ndmin=0)
        obj = np.zeros(code_value.shape, dtype=cls._dtype)
        obj["_code"] = code_value
        obj["_user"] = user_value
        obj = super().__new__(cls, obj, unit)
        return obj

    @property
    def code(self) -> float:
        return np.float64(self["_code"].view(np.ndarray))

    @code.setter
    def code(self, code_value: float):
        self["_code"] = code_value

    @property
    def user(self) -> Quantity:
        return self["_user"].view(Quantity)

    @user.setter
    def user(self, user_value: Quantity):
        self["_user"] = user_value

    def __str__(self) -> str:
        pfx = f"<{self.__class__.__name__} "
        sfx = f" {self.user.unit}>"
        blank_pfx = re.sub(".", " ", pfx)
        code_str = np.array2string(self.code, prefix=pfx + "code = ")
        user_str = np.array2string(self.user, prefix=pfx + "user = ")
        return f"{pfx}code = {code_str}\n{blank_pfx}user = {user_str}{sfx}"

    def __format__(self, format_spec) -> str:
        return self.__str__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise NotImplementedError

        c_args, u_args = [], []
        for input in inputs:
            if isinstance(input, Quantity):
                c_args.append(input.code)
                u_args.append(input.user)
            else:
                c_args.append(input)
                u_args.append(input)

        c_output = super().__array_ufunc__(ufunc, method, *c_args, **kwargs)
        u_output = super().__array_ufunc__(ufunc, method, *u_args, **kwargs)
        if c_output is NotImplemented or u_output is NotImplemented:
            raise NotImplementedError

        c_output = c_output.view(np.ndarray)
        u_output = u_output.view(Quantity)
        return SimQuantity(c_output, u_output.value, u_output.unit)

    def to(self, unit):
        out_user = self.user.to(unit)
        return SimQuantity(self.code, out_user.value, out_user.unit)

    def decompose(self):
        out_user = self.user.decompose()
        return SimQuantity(self.code, out_user.value, out_user.unit)

    def code_to_user(self, code_value: float) -> Quantity:
        return (code_value / self.code) * self.user

    def user_to_code(self, user_value: Quantity) -> float:
        return (user_value / self.user).decompose().value * self.code

    def check_consistency(self) -> None:
        assert isclose(self.code, self.user_to_code(self.user)).all()
        assert isclose(self.user, self.code_to_user(self.code)).all()
