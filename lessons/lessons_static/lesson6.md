
# Control Structures: File Handling in Python

## File Handling in Python

Python makes it easy to work with files on your computer. You can read from files and write to files.

## Opening Files

You use the `open()` function to open a file. It takes two main arguments: the filename and the mode.

Common modes:

* `"r"`: Read (default). Opens for reading. Error if the file does not exist.
* `"w"`: Write. Opens for writing. Creates the file if it does not exist, or **truncates** (empties) the file if it exists.
* `"a"`: Append. Opens for writing. Creates the file if it does not exist. Adds content to the end of the file.
* `"x"`: Create. Creates the specified file. Returns an error if the file exists.

It's best practice to use `with open(...) as file:` which ensures the file is properly closed even if errors occur.

## Writing to a file

```python
with open("my_file.txt", "w") as f:
    f.write("Hello, file world!\n")
    f.write("This is a new line.")

```

## Reading from a file

```python
with open("my_file.txt", "r") as f:
    content = f.read() # Read the entire content
    print(content)
