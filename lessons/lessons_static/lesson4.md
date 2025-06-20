
# Control Structures: Functions and Modules

## Functions and Modules

Functions are reusable blocks of code that perform a specific task. Modules are files containing Python definitions and statements, which you can import and use in your own code.

## Defining Functions

You define a function using the `def` keyword, followed by the function name, parentheses `()`, and a colon `:`.

```python
def greet():
    print("Hello, Python!")

greet() # Calling the function
A function can also take inputs, called parameters, inside its parentheses. These parameters act like variables specific to the function.

Python

def greet_person(name):
    print(f"Hello, {name}!")

greet_person("Alice") # Calling with an argument

```

Functions can also return 'values' using the return 'keywords'.

```python
def add(a, b):
    return a + b

result = add(5, 3)
print(result)
