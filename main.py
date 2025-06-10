import sys
import json
import os
import re
import traceback
from typing import Dict, List, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QTextEdit, QPushButton, QListWidget, QLineEdit,
    QLabel, QSizePolicy, QMessageBox
)
from PySide6.QtGui import QTextCharFormat, QSyntaxHighlighter, QTextDocument, QColor
from PySide6.QtCore import Qt, QRegularExpression

# Python Syntax Highlighter
class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.highlightingRules = []

        # Keywords
        keywordFormat = QTextCharFormat()
        keywordFormat.setForeground(QColor("#569cd6")) # Visual Studio Code blue
        keywords = [
            "False", "None", "True", "and", "as", "async", "await", "break",
            "class", "continue", "def", "del", "elif", "else", "except",
            "finally", "for", "from", "global", "if", "import", "in",
            "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
            "return", "try", "while", "with", "yield"
        ]
        self.highlightingRules.extend([(QRegularExpression(r"\b" + keyword + r"\b"), keywordFormat) for keyword in keywords])

        # Operators
        operatorFormat = QTextCharFormat()
        operatorFormat.setForeground(QColor("#d4d4d4")) # White/Light Gray
        operators = [
            r"=", r"==", r"!=", r"<", r"<=", r">", r">=",
            r"\+", r"-", r"\*", r"/", r"//", r"%", r"\*\*",
            r"=", r"\+=", r"-=", r"\*=", r"/=", r"%=", r"//=",
            r"\*\*=", r"&", r"\|", r"\^", r"~", r"<<", r">>",
            r"and", r"or", r"not"
        ]
        self.highlightingRules.extend([(QRegularExpression(operator), operatorFormat) for operator in operators])

        # Built-in functions
        builtinFormat = QTextCharFormat()
        builtinFormat.setForeground(QColor("#dcdcaa")) # Visual Studio Code yellow
        builtins = [
            "abs", "delattr", "hash", "memoryview", "set",
            "all", "dict", "help", "min", "setattr",
            "any", "dir", "hex", "next", "slice",
            "ascii", "divmod", "id", "object", "sorted",
            "bin", "enumerate", "input", "oct", "object", "sum",
            "bool", "eval", "int", "open", "pow", "super",
            "breakpoint", "exec", "isinstance", "ord", "str",
            "bytearray", "filter", "issubclass", "range", "vars",
            "callable", "format", "len", "list", "reversed", "__import__", "map", "classmethod", "getattr", "locals", "setattr", "staticmethod"
        ]
        self.highlightingRules.extend([(QRegularExpression(r"\b" + builtin + r"\b"), builtinFormat) for builtin in builtins])

        # Strings
        stringFormat = QTextCharFormat()
        stringFormat.setForeground(QColor("#ce9178")) # Visual Studio Code orange
        self.highlightingRules.append((QRegularExpression(r'".*?"'), stringFormat))
        self.highlightingRules.append((QRegularExpression(r"'.*?'"), stringFormat))

        # Comments
        commentFormat = QTextCharFormat()
        commentFormat.setForeground(QColor("#6a9955")) # Visual Studio Code green
        self.highlightingRules.append((QRegularExpression(r"#.*"), commentFormat))

        # Numbers
        numberFormat = QTextCharFormat()
        numberFormat.setForeground(QColor("#b5cea8")) # Visual Studio Code light green
        self.highlightingRules.append((QRegularExpression(r"\b[0-9]+\b"), numberFormat))

        # Classes (simple detection)
        classFormat = QTextCharFormat()
        classFormat.setForeground(QColor("#b8d7a3")) # Another light green
        self.highlightingRules.append((QRegularExpression(r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*"), classFormat))

        # Function definitions
        functionDefFormat = QTextCharFormat()
        functionDefFormat.setForeground(QColor("#dcdcaa")) # Visual Studio Code yellow
        self.highlightingRules.append((QRegularExpression(r"\bdef\s+[A-Za-z_][A-Za-z0-9_]*"), functionDefFormat))


    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            expression = QRegularExpression(pattern)
            it = expression.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)
        self.setCurrentBlockState(0)


class CodeEditor(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFontFamily("Consolas")
        self.setFontPointSize(12)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(' ')) # 4 spaces for tab
        self.highlighter = PythonHighlighter(self.document())
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e; /* Dark background like VS Code */
                color: #d4d4d4; /* Light text color */
                border: 1px solid #333333; /* Subtle border */
            }
        """)

class SelfImprovingAgent:
    def __init__(self):
        """Initialize the self-improving agent (rule-based)."""
        self.memory = {
            'successful_strategies': [],
            'failed_attempts': [],
            'learned_patterns': [],
            'performance_metrics': [],
        }
        self.capabilities = {
            'feedback_quality': 0.8,
            'error_detection': 0.7,
            'hint_relevance': 0.7,
        }
        self.iteration_count = 0

    def provide_feedback_on_code(self, user_code: str, problem_description: str,
                                 expected_output: Any = None, previous_errors: str = "",
                                 num_attempts: int = 1) -> str:
        """
        Provides intelligent, rule-based feedback on user's code for a specific problem.
        """
        feedback_messages = ["Agent Feedback (Rule-Based):"]
        
        # General encouraging message
        if num_attempts == 1:
            feedback_messages.append("Keep trying! Let's look for some clues.")
        elif num_attempts == 2:
            feedback_messages.append("Don't give up! Sometimes a fresh pair of eyes (or an agent's) helps.")
        elif num_attempts >= 3:
            feedback_messages.append("It seems like you're facing a tricky spot. Here are some more targeted suggestions:")

        # --- Rule-Based Checks for Common Python Beginner Mistakes ---

        # 1. Missing print statement if output is expected
        if expected_output is not None and "print(" not in user_code:
            feedback_messages.append("- Hint: If your goal is to produce specific output, are you using `print()` to display your result?")

        # 2. Basic Indentation Check (very rudimentary, needs improvement for complex cases)
        lines = user_code.split('\n')
        in_block = False
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if not stripped_line: # Skip empty lines
                continue

            # Check for missing colon after control flow or function definition
            if (stripped_line.startswith(("if ", "for ", "while ", "def ", "class ")) and
                not stripped_line.endswith(':') and
                not stripped_line.startswith('#')):
                feedback_messages.append(f"- Potential syntax error on line {i+1}: Statements like `if`, `for`, `while`, `def`, `class` usually end with a colon `:`. Check your syntax.")
                in_block = True
            elif stripped_line.endswith(':'):
                in_block = True
            elif in_block:
                current_indent = len(line) - len(line.lstrip())
                prev_line_indent = len(lines[i-1]) - len(lines[i-1].lstrip()) if i > 0 and lines[i-1].strip() else current_indent
                
                if i > 0 and lines[i-1].strip().endswith(':') and current_indent <= prev_line_indent:
                    feedback_messages.append(f"- Indentation hint on line {i+1}: The line after a statement ending with a colon (like `if`, `def`, `for`) must be indented. Remember indentation defines code blocks in Python.")
                in_block = False

        # 3. Using Built-in Names as Variables (simple string check)
        python_keywords = ["list", "dict", "str", "int", "float", "sum", "len", "max", "min"]
        for keyword in python_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\s*=\s*', user_code):
                feedback_messages.append(f"- Naming convention: You're using `{keyword}` as a variable name. It's a built-in Python function/type, so using it as a variable can lead to unexpected errors. Try a different name.")
                break

        # 4. Basic type of problem hints based on keywords in description
        problem_lower = problem_description.lower()
        if "loop" in problem_lower or "iterate" in problem_lower:
            if "for " not in user_code and "while " not in user_code:
                feedback_messages.append("- Consider using a loop (`for` or `while`) if you need to repeat an action or go through items in a collection.")
        if "function" in problem_lower or "def" in problem_lower:
            if "def " not in user_code:
                feedback_messages.append("- If you need to define a reusable block of code, remember to use the `def` keyword to define a function.")
        if "conditional" in problem_lower or "if" in problem_lower or "else" in problem_lower:
            if "if " not in user_code:
                feedback_messages.append("- To execute code based on a condition, you'll typically use an `if` statement.")
        if "list" in problem_lower or "array" in problem_lower:
            if "[" not in user_code and "list(" not in user_code:
                feedback_messages.append("- When working with collections of items, Python lists (created with `[]`) are very useful.")
        if "dictionary" in problem_lower or "dict" in problem_lower:
            if "{" not in user_code and "dict(" not in user_code:
                feedback_messages.append("- For key-value pairs, Python dictionaries (created with `{}`) are the go-to data structure.")


        # If only the initial generic message, add a simple default hint
        if len(feedback_messages) == 1:
            feedback_messages.append("Try reviewing the exercise prompt carefully. Pay attention to exactly what the problem asks you to do and what output it expects.")
            
        return "\n".join(feedback_messages)

    def analyze_task(self, task: str) -> Dict[str, Any]:
        """A placeholder for task analysis (rule-based)."""
        return {
            "complexity": 5,
            "required_skills": ["Python basics", "problem solving"],
            "potential_challenges": ["syntax errors", "logical errors"],
            "recommended_approach": "Break down the problem, write small pieces of code, test often.",
            "success_criteria": "Code produces correct output."
        }

    def learn_from_experience(self, success: bool, user_code: str, problem_id: str, feedback_given: str):
        """
        Conceptual learning: could log successes/failures to a file for later analysis,
        or update internal statistics about common errors.
        """
        if success:
            self.memory['successful_strategies'].append({"problem_id": problem_id, "code_length": len(user_code)})
        else:
            self.memory['failed_attempts'].append({"problem_id": problem_id, "code_snapshot": user_code, "feedback": feedback_given})


class PythonLearningTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Learning Tool for Beginners")
        self.setGeometry(100, 100, 1200, 800)

        self.current_lesson_index = 0
        self.current_exercise_index = 0
        self.base_lesson_content = ""
        self.exercise_attempts = {}

        self.init_ui()

        self.agent = SelfImprovingAgent() # Initialize the rule-based agent

        self.lessons_data = self.load_lessons_data("lessons/lessons.json")

        for i, lesson in enumerate(self.lessons_data):
            self.lesson_list.addItem(f"{i+1}. {lesson['title']}")
        self.lesson_list.setCurrentRow(0)

        self.load_lesson_content(self.current_lesson_index)

    def load_lessons_data(self, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.output_text_edit.setText(f"Error: lessons.json not found at {json_path}")
            return []
        except json.JSONDecodeError:
            self.output_text_edit.setText(f"Error: Could not decode lessons.json. Check its syntax.")
            return []

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel for lesson list
        left_panel = QVBoxLayout()
        self_lessons_label = QLabel("Lessons:")
        self_lessons_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_panel.addWidget(self_lessons_label)

        self.lesson_list = QListWidget()
        self.lesson_list.currentRowChanged.connect(self.lesson_selected)
        left_panel.addWidget(self.lesson_list)
        left_panel.addStretch(1)

        main_layout.addLayout(left_panel, 1)

        # Right panel for lesson content, code editor, and output
        right_panel = QVBoxLayout()

        # Lesson Content (top window)
        lesson_content_label = QLabel("Lesson Content:")
        lesson_content_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_panel.addWidget(lesson_content_label)

        self.lesson_content_display = QTextEdit()
        self.lesson_content_display.setReadOnly(True)
        self.lesson_content_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lesson_content_display.setStyleSheet("""
            QTextEdit {
                font-family: 'Segoe UI', sans-serif;
                font-size: 11pt;
                background-color: #f0f0f0;
                color: #333333;
                padding: 10px;
                border: 1px solid #cccccc;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #005f87;
                margin-top: 1em;
                margin-bottom: 0.5em;
            }
            pre {
                background-color: #e8e8e8;
                border-left: 3px solid #569cd6;
                padding: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                line-height: 1.4;
                margin-top: 1em;
                margin-bottom: 1em;
            }
            code {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                color: #c7254e;
            }
            strong {
                font-weight: bold;
                color: #2b91af;
            }
            p {
                margin-bottom: 0.5em;
            }
            ul, ol {
                margin-left: 1.5em;
            }
        """)
        right_panel.addWidget(self.lesson_content_display, 3)

        # Your Code (middle window)
        your_code_label = QLabel("Your Code:")
        your_code_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_panel.addWidget(your_code_label)

        self.code_editor = CodeEditor()
        self.code_editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel.addWidget(self.code_editor, 2)

        # Buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Code")
        self.run_button.clicked.connect(self.run_code)
        button_layout.addWidget(self.run_button)

        self.check_button = QPushButton("Check Answer")
        self.check_button.clicked.connect(self.check_answer)
        button_layout.addWidget(self.check_button)
        right_panel.addLayout(button_layout)

        # Output (bottom window)
        output_label = QLabel("Output:")
        output_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_panel.addWidget(output_label)

        self.output_text_edit = QTextEdit()
        self.output_text_edit.setReadOnly(True)
        self.output_text_edit.setFontFamily("Consolas")
        self.output_text_edit.setFontPointSize(10)
        self.output_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.output_text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #00ff00;
                border: 1px solid #333333;
            }
        """)
        right_panel.addWidget(self.output_text_edit, 1)

        main_layout.addLayout(right_panel, 4)

    def lesson_selected(self, index):
        self.current_lesson_index = index
        self.current_exercise_index = 0
        self.load_lesson_content(index)
        lesson_id = self.lessons_data[self.current_lesson_index].get('id', f"lesson_{self.current_lesson_index}")
        for exercise_idx in range(len(self.lessons_data[index].get('exercise_checks', []))):
            key = f"{lesson_id}-{exercise_idx}"
            self.exercise_attempts[key] = 0

    def load_lesson_content(self, index):
        if 0 <= index < len(self.lessons_data):
            lesson = self.lessons_data[index]
            md_file_path = os.path.join(os.path.dirname(__file__), lesson['filename'])

            try:
                with open(md_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                self.base_lesson_content = content
                self.update_exercise_prompt_in_display()

                current_exercise_data = lesson.get('exercise_checks', [])
                if current_exercise_data and self.current_exercise_index < len(current_exercise_data):
                    initial_code_for_current_exercise = current_exercise_data[self.current_exercise_index].get('initial_code', '')
                    self.code_editor.setText(initial_code_for_current_exercise)
                else:
                    self.code_editor.setText(lesson.get('initial_code', ''))

            except FileNotFoundError:
                self.output_text_edit.setText(f"Error: Lesson file not found: `{md_file_path}`")
                self.lesson_content_display.setMarkdown(f"Error: Lesson file not found: `{md_file_path}`")
                self.base_lesson_content = ""
            except Exception as e:
                error_message = f"Error loading lesson content: {e}"
                self.output_text_edit.setText(error_message)
                self.lesson_content_display.setMarkdown(f"Error loading lesson content: {e}")
                self.base_lesson_content = ""
        else:
            self.lesson_content_display.setMarkdown("No lesson selected or lesson index out of bounds.")
            self.output_text_edit.setText("No lesson selected or lesson index out of bounds.")
            self.base_lesson_content = ""

    def update_exercise_prompt_in_display(self):
        combined_text = self.base_lesson_content

        if 0 <= self.current_lesson_index < len(self.lessons_data):
            current_lesson = self.lessons_data[self.current_lesson_index]
            exercises = current_lesson.get('exercise_checks', [])

            if 0 <= self.current_exercise_index < len(exercises):
                prompt_text = exercises[self.current_exercise_index]['prompt']
                combined_text += f"\n\n### Current Exercise:\n{prompt_text}"
            else:
                combined_text += "\n\nAll exercises for this lesson are complete!"
        else:
            combined_text += "\n\nNo active lesson or exercise prompt available."

        if hasattr(self, 'lesson_content_display'):
            self.lesson_content_display.setMarkdown(combined_text)
            self.lesson_content_display.verticalScrollBar().setValue(0)
        else:
            print("Error: lesson_content_display not found during update_exercise_prompt_in_display")


    def run_code(self):
        code = self.code_editor.toPlainText()
        self.output_text_edit.setText("Running code...\n")
        try:
            old_stdout = sys.stdout
            sys.stdout = OutputRedirector(self.output_text_edit)

            exec_globals = {}
            exec_locals = {}

            exec(code, exec_globals, exec_locals)

            sys.stdout = old_stdout
            self.output_text_edit.append("\nCode execution finished.")

        except Exception as e:
            sys.stdout = old_stdout
            tb_str = traceback.format_exc()
            self.output_text_edit.append(f"\nErrors:\n{tb_str}")


    def check_answer(self):
        current_lesson = self.lessons_data[self.current_lesson_index]
        exercises = current_lesson.get('exercise_checks', [])

        if not exercises:
            self.output_text_edit.setText("No exercises to check for this lesson.")
            return

        if self.current_exercise_index >= len(exercises):
            self.output_text_edit.setText("All exercises for this lesson are complete!")
            return

        current_exercise = exercises[self.current_exercise_index]
        user_code = self.code_editor.toPlainText()
        check_function_str = current_exercise['check_function']
        expected_output = current_exercise.get('expected_output', None)
        problem_description = current_exercise['prompt']

        lesson_id = self.lessons_data[self.current_lesson_index].get('id', f"lesson_{self.current_lesson_index}")
        exercise_id = self.current_exercise_index
        full_exercise_key = f"{lesson_id}-{exercise_id}"
        self.exercise_attempts[full_exercise_key] = self.exercise_attempts.get(full_exercise_key, 0) + 1
        num_attempts = self.exercise_attempts[full_exercise_key]


        self.output_text_edit.setText("Checking answer...\n")

        check_exec_context = {} 
        old_stdout = sys.stdout
        output_buffer = []
        sys.stdout = type('stdout', (object,), {'write': lambda self, x: output_buffer.append(x)})()

        check_result_bool = False
        execution_errors = ""

        try:
            exec(check_function_str, check_exec_context)
            
            if 'check_result' in check_exec_context and callable(check_exec_context['check_result']):
                check_result_func = check_exec_context['check_result']
                check_result_bool = check_result_func(user_code, expected_output)
            else:
                execution_errors = "Error: 'check_result' function not found or not callable within the check string."
                self.output_text_edit.append(execution_errors)
                check_result_bool = False

        except Exception as e:
            execution_errors = f"Error during check execution: {traceback.format_exc()}"
            self.output_text_edit.append(execution_errors)
            check_result_bool = False
        finally:
            sys.stdout = old_stdout
            captured_output_str = "".join(output_buffer)
            if captured_output_str:
                self.output_text_edit.append(f"Output during check:\n{captured_output_str.strip()}")

        if check_result_bool:
            self.output_text_edit.append("Correct! Moving to next exercise.")
            self.current_exercise_index += 1
            self.update_exercise_prompt_in_display()
            self.exercise_attempts[full_exercise_key] = 0

            self.agent.learn_from_experience(True, user_code, full_exercise_key, "")

            if 0 <= self.current_lesson_index < len(self.lessons_data):
                lesson = self.lessons_data[self.current_lesson_index]
                exercises_in_lesson = lesson.get('exercise_checks', [])
                if self.current_exercise_index < len(exercises_in_lesson):
                    self.code_editor.setText(exercises_in_lesson[self.current_exercise_index].get('initial_code', ''))
                else:
                    self.code_editor.setText(lesson.get('initial_code', ''))

            if self.current_exercise_index >= len(exercises):
                self.output_text_edit.append("Congratulations! You've completed all exercises for this lesson.")
        else:
            self.output_text_edit.append("Incorrect. Please try again.")
            
            # --- Rule-based feedback is now active ---
            feedback_message = self.agent.provide_feedback_on_code(
                user_code=user_code,
                problem_description=problem_description,
                expected_output=expected_output,
                previous_errors=execution_errors + "\n" + captured_output_str,
                num_attempts=num_attempts
            )
            self.output_text_edit.append(f"\n--- Agent Feedback (Rule-Based) ---\n{feedback_message}")
            # --- End rule-based feedback ---

            self.agent.learn_from_experience(False, user_code, full_exercise_key, feedback_message)


class OutputRedirector:
    def __init__(self, text_edit):
        self.text_edit = text_edit
        self.buffer = []

    def write(self, text):
        self.buffer.append(text)
        self.text_edit.insertPlainText(text)
        QApplication.processEvents()

    def flush(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PythonLearningTool()
    window.show()
    sys.exit(app.exec())