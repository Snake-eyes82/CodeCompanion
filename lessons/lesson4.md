
# Control Structures: `Functions and Modules`**

```markdown
# Functions and Modules

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
Functions can also return values using the return keyword.

Python

def add(a, b):
    return a + b

result = add(5, 3)
print(result) # Output: 8
Exercise 1: Simple Addition Function
Task: Define a function add(a, b) that takes two parameters, a and b, and returns their sum.
Instructions:

Define the function add with two parameters.
Inside the function, return the sum of a and b.
Call add(7, 3) and print the result. Example Output:
10
Exercise 2: Modifying a Greeting Function
Task: Modify the greet() function to accept a name parameter and print a personalized greeting.
Instructions:

Change the definition of greet() to greet(name).
Inside the function, print "Hello, [name]!" (replace [name] with the value of the name parameter).
Call the modified greet() function with your name (e.g., greet('Alice')). Example Output:
Hello, Alice!
