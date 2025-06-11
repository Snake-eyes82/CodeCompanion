
# Control Structures: If Statements and Loops

Control structures allow you to control the flow of your program. They decide *when* and *how many times* certain blocks of code should execute.

## If Statements

An `if` statement allows you to execute code only if a certain condition is true.

```python
x = 10
if x > 5:
    print("x is greater than 5")

```

### Elif and Else Clauses

Often, you have multiple conditions to check. Python provides elif (short for "else if") and else clauses to handle these scenarios.

An elif block is checked only if the preceding if or elif conditions were False.
The else block is a catch-all; its code runs only if none of the preceding if or elif conditions were True. Only one else block is allowed.

#### Example with elif and else

```python

score = 85
if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
else:
    print("Grade: C or lower")

```

### For Loops

A for loop is used for iterating over a sequence (like a list, tuple, string, or a range of numbers). It repeatedly executes a block of code for each item in that sequence. This is ideal when you know in advance how many times you want to loop, or when you want to process every item in a collection.

The range() function is very commonly used with for loops to generate a sequence of numbers. For example, range(5) generates numbers 0, 1, 2, 3, and 4.

#### Example 1: Iterating a specific number of times with range():**

This loop will run 5 times.
In each iteration, 'i' will take the values 0, then 1, then 2, then 3, then 4.

```python

for i in range(5):
    print(i)

```

#### Example 2: Iterating over items in a list:**

```python

fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

This loop will run once for each item in the 'fruits' list.
In each iteration, 'fruit' will be assigned the current item.

```

### While Loops

A while loop repeatedly executes a block of code as long as a given condition remains True. It's crucial to include code within the loop that eventually makes the condition False, otherwise, your program will get stuck in an infinite loop (it will never stop running!).

#### Example:**

```python

count = 0
while count < 3: # The loop continues as long as 'count' is less than 3
    print("Count is:", count)
    count += 1 # This line increments 'count', eventually making the condition 'count < 3' False
