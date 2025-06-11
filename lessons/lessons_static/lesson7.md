
# Control Structures: Error Handling and Exceptions

## Error Handling and Exceptions

Errors happen! In Python, when an error occurs during execution, it's called an **exception**. You can "handle" these exceptions gracefully to prevent your program from crashing.

## Try-Except Blocks

The `try-except` block is used to catch and handle exceptions.

```python
try:
    # Code that might raise an error
    numerator = 10
    denominator = 0
    result = numerator / denominator # This will cause a ZeroDivisionError
    print(result)
except ZeroDivisionError:
    # Code to execute if a ZeroDivisionError occurs
    print("Error: Cannot divide by zero!")
