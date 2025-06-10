
# Control Structures: `Object-Oriented Programming Basics`

```markdown
# Object-Oriented Programming (OOP) Basics

Object-Oriented Programming (OOP) is a programming paradigm based on the concept of "objects", which can contain data (attributes) and code (methods).

## Classes and Objects

* A **class** is a blueprint or a template for creating objects.
* An **object** (or instance) is a specific realization of a class.

## Defining a Class

You define a class using the `class` keyword.

```python
class Dog:
    # Class attribute (shared by all instances)
    species = "Canis familiaris"

    # The __init__ method is a constructor, called when a new object is created
    def __init__(self, name, breed):
        # Instance attributes (unique to each object)
        self.name = name
        self.breed = breed
        self.tricks = [] # A list unique to each dog

    # Instance method
    def bark(self):
        return f"{self.name} says Woof!"

    def add_trick(self, trick):
        self.tricks.append(trick)
        print(f"{self.name} learned {trick}!")
