def add(a, b):
    return a + b


def add(a: int, b: int) -> int:
    return a + b

# The difference between the first code and the last code is that
# python doesn't care the input is int or output is int or not.
# While the second one does.

# Type hint functions are note to humans and (tools) and about what kind of data
# or variable or functions to use. It doesn't change how Python runs it. It is
# only for clarity and checking, not enforcement

# The thing, even if the variables are not int they will still run.