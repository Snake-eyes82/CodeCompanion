import sys
import json
import os
import re
import traceback
from typing import Dict, List, Any

# --- Gemini API Integration ---
import google.generativeai as genai
from google.api_core import exceptions

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QTextEdit, QPushButton, QListWidget, QLineEdit,
    QLabel, QSizePolicy, QMessageBox,
    QSplitter
)
from PySide6.QtGui import QTextCharFormat, QSyntaxHighlighter, QTextDocument, QColor
from PySide6.QtCore import Qt, QRegularExpression

# Python Syntax Highlighter (NO CHANGE)
class PythonSyntaxHighlighter(QSyntaxHighlighter):
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
            r"&", r"\|", r"\^", r"~", r"<<", r">>",
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


# Code Editor (NO CHANGE)
class CodeEditor(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFontFamily("Consolas")
        self.setFontPointSize(12)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(' ')) # 4 spaces for tab
        self.highlighter = PythonSyntaxHighlighter(self.document())
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e; /* Dark background like VS Code */
                color: #d4d4d4; /* Light text color */
                border: 1px solid #333333; /* Subtle border */
            }
        """)


class SelfImprovingAgent:
    def __init__(self, api_key=None):
        """Initialize the agent with Gemini API, loading key from environment variable."""
        # FIX: Consolidated API key initialization to use GOOGLE_API_KEY as standard
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        if not self.api_key:
            print("Error: GOOGLE_API_KEY environment variable not set or API key not provided.")
            print("Please set it before running the application for advanced features.")
            self.model = None
            self.chat = None # Ensure chat is also None if model is None
            self.api_status = "API_KEY_MISSING"
        else:
            try:
                genai.configure(api_key=self.api_key)
                # Using 'gemini-1.5-flash' as a good balance for educational tools
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.chat = self.model.start_chat(history=[]) # Initialize chat here
                print("Gemini model initialized successfully.")
                self.api_status = "READY"
            except Exception as e:
                self.model = None
                self.chat = None
                self.api_status = "API_INIT_FAILED"
                print(f"Failed to initialize Gemini model: {e}")
                print("Feedback will be limited to rule-based or unavailable.")

        self.memory = {
            'successful_strategies': [],
            'failed_attempts': [],
            'learned_patterns': [],
            'performance_metrics': [],
        }
        self.capabilities = {
            'feedback_quality': 0.8, # Placeholder, actual quality from API
            'error_detection': 0.7,
            'hint_relevance': 0.7,
        }
        self.iteration_count = 0

    def _generate_content_with_error_handling(self, prompt_parts: List[str], **kwargs) -> str:
        """Helper to call Gemini API with robust error handling."""

        if self.api_status == "API_KEY_MISSING":
            return "Agent Feedback: Gemini API key not found. Please set the GOOGLE_API_KEY environment variable for advanced features."
        if self.api_status == "API_INIT_FAILED" or not hasattr(self, 'model') or not self.model:
            return "Agent Feedback: Gemini API could not be initialized. Check your internet connection or API key validity."

        default_generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 800,
        }

        default_safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        try:
            final_generation_config = {
                **default_generation_config,
                **kwargs.get('generation_config', {})
            }
            final_safety_settings = kwargs.get('safety_settings', default_safety_settings)

            response = self.model.generate_content(
                prompt_parts,
                generation_config=final_generation_config,
                safety_settings=final_safety_settings
            )

            if response and response.text:
                return response.text
            elif response and response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                print(f"Prompt was blocked by safety settings: {block_reason}")
                return f"Your request was blocked by safety settings ({block_reason}). Please try rephrasing."
            else:
                return "Could not generate a response (empty or unexpected API response)."

        except exceptions.ResourceExhausted as e:
            print(f"API Rate Limit Exceeded: {e}")
            return f"API Rate Limit Exceeded. Please wait a moment and try again. ({e})"
        except exceptions.FailedPrecondition as e:
            if "API key not valid" in str(e):
                print(f"Invalid API Key: {e}")
                return f"Invalid API Key. Please double-check your GOOGLE_API_KEY environment variable. ({e})"
            else:
                print(f"Gemini API Precondition Failed: {e}")
                return f"A Gemini API error occurred. Check input or model status. ({e})"
        except exceptions.GoogleAPIError as e:
            print(f"Gemini API Error: {e}")
            return f"A Gemini API error occurred. Please try again later. ({e})"
        except Exception as e:
            print(f"Unexpected error calling Gemini API: {e}")
            return f"An unexpected error occurred while generating feedback. ({e})"


    def provide_feedback_on_code(self, user_code: str, problem_description: str,
                                 expected_output: Any = None, previous_errors: str = "",
                                 num_attempts: int = 1) -> str:
        """Provides intelligent feedback using the Gemini API."""
        prompt_parts = [
            f"You are an expert Python programming tutor helping a beginner. The user is attempting to solve the following problem:\n",
            f"Problem Description: {problem_description}\n",
            f"Expected Output (if applicable): {expected_output}\n",
            f"Here is the user's current code:\n```python\n{user_code}\n```\n",
        ]

        if previous_errors:
            prompt_parts.append(f"The user's code produced the following errors or output when executed:\n```\n{previous_errors}\n```\n")

        prompt_parts.append(f"Given this, provide constructive and encouraging feedback. Point out any errors, suggest improvements, and offer hints without giving away the direct answer. Focus on Python syntax, common pitfalls for beginners, and logical flow. If the code is correct, congratulate them.\n\nFeedback:")

        feedback = self._generate_content_with_error_handling(prompt_parts)
        self.learn_from_experience(False, user_code, "feedback_request", feedback)
        return f"Agent Feedback:\n{feedback}"

    # General AI Question (NO CHANGE)
    def ask_general_question(self, user_query: str) -> str:
        """
        Allows the user to ask the AI general questions about Python or the current task.
        """
        prompt = (
            "You are an expert Python programming tutor. The user has a question "
            "about Python, programming concepts, or their current task. "
            "Provide a helpful and clear answer. Keep it concise but comprehensive. "
            f"User's question: {user_query}\n\nAnswer:"
        )
        response_text = self._generate_content_with_error_handling([prompt])
        return f"AI Response:\n{response_text}"


    def analyze_task(self, task: str) -> Dict[str, Any]:
        """A placeholder for task analysis (would be rule-based or human-curated)."""
        return {
            "complexity": 5,
            "required_skills": ["Python basics", "problem solving"],
            "potential_challenges": ["syntax errors", "logical errors"],
            "recommended_approach": "Break down the problem, write small pieces of code, test often.",
            "success_criteria": "Code produces correct output."
        }

    def learn_from_experience(self, success: bool, user_code: str, problem_id: str, feedback_given: str):
        """
        Conceptual learning: logs successes/failures. For a real product, this would
        involve more sophisticated data storage and analysis.
        """
        if success:
            self.memory['successful_strategies'].append({"problem_id": problem_id, "code_length": len(user_code)})
        else:
            self.memory['failed_attempts'].append({"problem_id": problem_id, "code_snapshot": user_code, "feedback": feedback_given})

    def generate_lesson_content(self, topic: str, difficulty: str = "beginner", num_exercises: int = 2) -> Dict[str, Any]:
        """
        Generates a full lesson, including markdown content and exercises.
        Updated prompt for more explicit JSON output and added robust parsing.
        """
        prompt = [
            f"You are an expert Python programming tutor. Generate a comprehensive Python lesson on the topic of '{topic}' "
            f"for a {difficulty} level programmer. The lesson should be engaging, clear, and include a main concept, "
            f"examples, and explanations.",
            "After the lesson content, generate a section titled 'Exercises:'. "
            f"For each exercise, provide:\n"
            f"1. A 'prompt' (the exercise description).\n"
            f"2. 'initial_code' (optional starting code for the user, usually empty or with a function signature).\n"
            f"3. 'check_function' (a Python function string named 'check_result(user_code, expected_output)' that "
            f"evaluates the user's code. This function should print messages indicating success or failure. It "
            f"must return True for False for failure. It should use `exec` with a controlled `globals()` "
            f"to run user_code and then assert conditions. Be very careful to make this function robust and self-contained). Example `check_function` structure:\n"
            f"```python\n"
            f"def check_result(user_code, expected_output):\n"
            f"    local_output = []\n"
            f"    try:\n"
            f"        exec_globals = {{'__builtins__': {{'print': lambda *args, **kwargs: local_output.append(' '.join(map(str, args)))}}}}\n"
            f"        exec(user_code, exec_globals)\n"
            f"        # Add your specific check logic here based on local_output or other conditions\n"
            f"        if 'Hello, World!' in ''.join(local_output):\n"
            f"            print('Test passed: Output contains \"Hello, World!\".')\n"
            f"            return True\n"
            f"        else:\n"
            f"            print(f'Test failed: Expected \"Hello, World!\", got {{''.join(local_output)}}')\n"
            f"            return False\n"
            f"    except Exception as e:\n"
            f"        print(f'Code execution error: {{e}}')\n"
            f"        return False\n"
            f"```\n"
            f"4. 'expected_output' (the expected console output from a correct solution, if applicable). Use the actual string, not a description.\n",
            # More explicit prompt for JSON format
            f"Generate {num_exercises} exercises. Format the *entire* output as a single JSON string. Do NOT include any text, preambles, or markdown formatting (like ```json) outside the JSON block. The JSON should start directly with '{{' and end with '}}'. Ensure all internal strings (especially for 'check_function') are properly JSON-escaped (e.g., newlines as \\n, double quotes as \\)."
        ]

        generation_config = {"temperature": 0.9, "max_output_tokens": 2000}
        response_text = self._generate_content_with_error_handling(prompt, generation_config=generation_config)
        
        # --- Start of JSON Extraction and Parsing Logic ---
        parsed_data = {}
        error_message = ""

        # Log the raw AI response for debugging
        print(f"\n--- Raw AI Response for '{topic}' (start) ---")
        print(response_text)
        print(f"--- Raw AI Response for '{topic}' (end) ---\n")

        try:
            # Attempt 1: Direct parse after stripping common markdown fences
            cleaned_response_text = response_text.strip()
            
            # Remove leading/trailing markdown code block fences if they exist
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[len("```json"):].strip()
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text.rstrip("```").strip()

            # Attempt to find the actual JSON object (from first { to last })
            first_brace = cleaned_response_text.find('{')
            last_brace = cleaned_response_text.rfind('}')

            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_candidate = cleaned_response_text[first_brace : last_brace + 1]
                print(f"Attempting to parse extracted JSON candidate:\n{json_candidate[:500]}...")
                parsed_data = json.loads(json_candidate)
            else:
                raise ValueError("No valid JSON object delimiters found in the response.")

        except (json.JSONDecodeError, ValueError) as e:
            error_message = f"Error decoding JSON from AI (Attempt 1): {e}. Trying fallback regex."
            print(error_message)
            print(f"Problematic JSON candidate: {json_candidate[:500]}...") # Log what failed to parse

            # Attempt 2: Fallback regex search for anything resembling a JSON block
            json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
            if json_match:
                try:
                    re_matched_json_string = json_match.group(1).strip()
                    print(f"Successfully extracted JSON via regex fallback:\n{re_matched_json_string[:500]}...")
                    parsed_data = json.loads(re_matched_json_string)
                    error_message = "" # Clear error if successful
                except json.JSONDecodeError as re_e:
                    error_message = f"Error decoding JSON from regex match fallback (Attempt 2): {re_e}"
                    print(error_message)
                    print(f"Regex matched string that failed to parse: {re_matched_json_string[:500]}...")
            else:
                error_message = f"Error: Could not find a parseable JSON structure in the AI response even with regex fallback."
                print(error_message)
        except Exception as e:
            error_message = f"Unexpected error during AI response processing: {e}"
            print(error_message)
            print(f"Traceback: {traceback.format_exc()}")
        
        # --- Post-processing and Error Return ---
        if not parsed_data or error_message:
            return {
                "lesson_content_markdown": f"Error: Could not generate lesson content. {error_message} Raw response (truncated): {response_text[:500]}...",
                "exercises": []
            }

        # If parsing was successful, proceed with content cleaning
        if 'lesson_content_markdown' in parsed_data:
            parsed_data['lesson_content_markdown'] = parsed_data['lesson_content_markdown'].replace('\u00A0', ' ')
        
        if 'exercises' in parsed_data:
            for exercise in parsed_data['exercises']:
                if 'prompt' in exercise:
                    exercise['prompt'] = exercise['prompt'].replace('\u00A0', ' ')
                if 'initial_code' in exercise:
                    exercise['initial_code'] = exercise['initial_code'].replace('\u00A0', ' ')
                if 'check_function' in exercise:
                    check_func = exercise['check_function']
                    check_func = check_func.replace('\u00A0', ' ') # Replace non-breaking space
                    exercise['check_function'] = check_func
        
        return parsed_data


    def generate_exercise_solution_check(self, user_code: str, problem_description: str, expected_output: str) -> str:
        """
        Generates a `check_function` string for a given problem and expected output.
        This can be used to dynamically create check functions.
        """
        prompt = [
            f"You are a Python programming expert. Given the following problem description, user's code, and expected output, "
            f"generate a Python function string named 'check_result(user_code, expected_output)' that evaluates if the "
            f"user's code correctly solves the problem. The function should print messages indicating success or failure. "
            f"It must return True for success and False for failure. It should use `exec` with a controlled `globals()` "
            f"to run user_code and then assert conditions. Ensure the check is robust and covers edge cases if possible.\n"
            f"Problem Description: {problem_description}\n"
            f"Expected Output: {expected_output}\n"
            f"User's (reference) code: {user_code}\n"
            f"Generate ONLY the Python function string, no extra text or markdown outside the function."
        ]
        # Lower temperature for more factual/direct generation of code
        generation_config = {"temperature": 0.5, "max_output_tokens": 500}
        return self._generate_content_with_error_handling(prompt, generation_config=generation_config)


class PythonLearningTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Learning Tool for Beginners")
        self.setGeometry(100, 100, 1200, 800)
        
        # --- ADD THESE LINES FOR DEBUGGING ---
        print(f"DEBUG: os.getcwd() = {os.getcwd()}")
        print(f"DEBUG: os.path.dirname(__file__) = {os.path.dirname(__file__)}")
        # -------------------------------------

        self.current_lesson_index = 0
        self.current_exercise_index = 0
        self.base_lesson_content = "" # This will now be set by AI or local file
        self.current_exercises_data = [] # This will now be set by AI or local file
        self.exercise_attempts = {}

        self.init_ui()

        self.agent = SelfImprovingAgent()

        # Ensure 'lessons' directory exists before attempting to load or save
        lessons_dir = os.path.join(os.path.dirname(__file__), "lessons")
        os.makedirs(lessons_dir, exist_ok=True)

        # Load lessons data first. This acts as our initial source for topics and offline content.
        self.lessons_data = self.load_lessons_data(os.path.join(lessons_dir, "lessons.json"))

        # Populate lesson_topics from the loaded lessons_data titles.
        if self.lessons_data:
            self.lesson_topics = [lesson['title'] for lesson in self.lessons_data]
        else:
            self.lesson_topics = [
                "Introduction to Python: Hello World!",
                "Variables and Data Types",
                "Control Structures: If Statements and Loops",
                "Functions and Modules",
                "Data Structures: Lists, Tuples, and Dictionaries",
                "File Handling in Python",
                "Error Handling and Exceptions",
                "Object-Oriented Programming Basics"
            ]
            self.output_text_edit.append("\nWarning: 'lessons.json' was empty or not found. Using default lesson topics.")
            print(f"DEBUG: lessons.json not found or empty. Using default topics.")

        for i, topic in enumerate(self.lesson_topics):
            self.lesson_list.addItem(f"{i+1}. {topic}")

        # Automatically select the first lesson to generate/load its content
        if self.lesson_topics:
            self.lesson_list.setCurrentRow(0)
        else:
            self.output_text_edit.append("\nNo lesson topics available to load.")

    def load_lessons_data(self, json_path):
        """
        Loads lessons data from a JSON file.
        Made more robust for FileNotFoundError and JSONDecodeError.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.output_text_edit.append(f"Info: '{json_path}' not found. Starting with empty lessons data.")
            return []
        except json.JSONDecodeError:
            self.output_text_edit.append(f"Error: Could not decode '{json_path}'. Check its syntax. Starting with empty lessons data.")
            return []
        except Exception as e:
            self.output_text_edit.append(f"An unexpected error occurred loading lessons data: {e}")
            return []
                
    def save_lesson_to_file(self, lesson_index: int, lesson_title: str,
                            lesson_markdown: str, exercises: List[Dict[str, Any]]):
        """
        Saves the generated lesson content to a markdown file and updates the lessons.json.
        Ensures lessons_generated directory exists and updates lesson_topics/QListWidget if new lesson.
        """
        # Generated lessons will use the underscore filename pattern
        lesson_filename = f"lesson_{lesson_index+1}.md"

        lessons_generated_dir = os.path.join(os.path.dirname(__file__), "lessons", "lessons_generated")
        md_file_path = os.path.join(lessons_generated_dir, lesson_filename)
        
        json_file_path = os.path.join(os.path.dirname(__file__), "lessons", "lessons.json")

        os.makedirs(lessons_generated_dir, exist_ok=True)

        if not lesson_markdown or "Error: Could not generate lesson content" in lesson_markdown:
            self.output_text_edit.append(f"Skipping saving '{lesson_title}' to MD: Invalid or empty content received from AI.")
            print(f"DEBUG SAVE: Skipping MD save for '{lesson_title}' due to invalid content.") # Added debug
            return

        try:
            # --- ADD THESE DEBUG PRINTS ---
            print(f"DEBUG SAVE: Attempting to write MD to: '{md_file_path}'")
            print(f"DEBUG SAVE: Length of markdown content: {len(lesson_markdown)} characters.")
            if len(lesson_markdown) > 500: # Print first 500 chars if content is long
                print(f"DEBUG SAVE: First 500 chars of content:\n{lesson_markdown[:500]}")
            else: # Print full content if short
                print(f"DEBUG SAVE: Full markdown content:\n{lesson_markdown}")
            # ------------------------------

            # Save Markdown content to file
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(lesson_markdown)
            self.output_text_edit.append(f"Markdown content for '{lesson_title}' saved to '{md_file_path}'.")
            print(f"DEBUG SAVE: Successfully wrote MD content to '{md_file_path}'.") # Added debug

            # Prepare the updated lesson entry for lessons.json
            updated_lesson_entry = {
                "title": lesson_title,
                "filename": os.path.join("lessons", "lessons_generated", lesson_filename), 
                "exercise_checks": exercises
            }

            # Update or add the lesson entry in lessons_data list
            found_existing = False
            for i, entry in enumerate(self.lessons_data):
                if entry.get('title') == lesson_title:
                    self.lessons_data[i] = updated_lesson_entry
                    found_existing = True
                    break
            
            if not found_existing:
                self.lessons_data.append(updated_lesson_entry)
                self.lesson_topics.append(lesson_title)
                self.lesson_list.addItem(f"{len(self.lesson_topics)}. {lesson_title}")
                print(f"DEBUG SAVE: Added new lesson '{lesson_title}' to internal data and UI list.") # Added debug


            # Save the updated lessons_data back to lessons.json
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.lessons_data, f, indent=4, ensure_ascii=False)
            self.output_text_edit.append(f"Lessons metadata for '{lesson_title}' updated in '{json_file_path}'.")
            print(f"DEBUG SAVE: Successfully updated lessons.json at '{json_file_path}'.") # Added debug

        except IOError as e:
            self.output_text_edit.append(f"Error writing lesson file '{md_file_path}': {e}")
            print(f"DEBUG SAVE: IOError caught: {e}") # Added debug
        except json.JSONDecodeError as e:
            self.output_text_edit.append(f"Error encoding JSON for '{json_file_path}': {e}")
            self.output_text_edit.append(f"Problematic exercise data: {exercises}") 
            print(f"DEBUG SAVE: JSONDecodeError caught: {e}") # Added debug
        except Exception as e:
            self.output_text_edit.append(f"An unexpected error occurred saving lesson: {e}")
            print(f"DEBUG SAVE: Generic Exception caught: {e}") # Added debug
            
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main splitter (Horizontal)
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(main_splitter)
        central_widget.layout().setContentsMargins(0,0,0,0)

        # --- Left Panel ---
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_layout.setContentsMargins(5,5,5,5)

        lessons_label = QLabel("Lessons:")
        lessons_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_panel_layout.addWidget(lessons_label)

        self.lesson_list = QListWidget()
        self.lesson_list.currentRowChanged.connect(self.lesson_selected)
        left_panel_layout.addWidget(self.lesson_list)
        left_panel_layout.setStretchFactor(self.lesson_list, 1)

        output_label = QLabel("Output:")
        output_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_panel_layout.addWidget(output_label)

        self.output_text_edit = QTextEdit()
        self.output_text_edit.setReadOnly(True)
        self.output_text_edit.setFontFamily("Consolas")
        self.output_text_edit.setFontPointSize(10)
        self.output_text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #00ff00;
                border: 1px solid #333333;
            }
        """)
        left_panel_layout.addWidget(self.output_text_edit)
        left_panel_layout.setStretchFactor(self.output_text_edit, 2)

        main_splitter.addWidget(left_panel_widget)

        # --- Right Panel (organized with vertical splitters) ---
        right_vertical_splitter = QSplitter(Qt.Vertical)

        # 1. Lesson Content Section
        lesson_content_widget = QWidget()
        lesson_content_layout = QVBoxLayout(lesson_content_widget)
        lesson_content_layout.setContentsMargins(5,5,5,5)

        # Static "Lesson Content:" heading
        lesson_content_label = QLabel("Lesson Content:")
        lesson_content_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        lesson_content_layout.addWidget(lesson_content_label)

        # Main QTextEdit for the base lesson markdown content
        self.lesson_content_display = QTextEdit()
        self.lesson_content_display.setReadOnly(True)
        self.lesson_content_display.setStyleSheet("""
            QTextEdit {
                font-family: 'Segoe UI', sans-serif;
                font-size: 11pt;
                background-color: #f0f0f0;
                color: #333333;
                padding: 10px;
                border: 1px solid #cccccc;
            }
            h1, h2, h3, h4, h5, h6 { /* Default styling for headings *in* the markdown content */
                color: #005f87;
                margin-top: 1em;
                margin-bottom: 0.5em;
                font-weight: bold;
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
        lesson_content_layout.addWidget(self.lesson_content_display)

        self.current_exercise_label = QLabel("")
        self.current_exercise_label.setStyleSheet("""
            QLabel {
                font-size: 18pt;
                font-weight: bold;
                color: #005f87;
                margin-top: 1.0em;
                margin-bottom: 0.2em;
            }
        """)
        self.current_exercise_label.hide()
        lesson_content_layout.addWidget(self.current_exercise_label)

        self.exercise_prompt_label = QLabel("")
        self.exercise_prompt_label.setWordWrap(True)
        self.exercise_prompt_label.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #333333;
                margin-top: 0.2em;
                margin-bottom: 1.0em;
            }
        """)
        self.exercise_prompt_label.hide()
        lesson_content_layout.addWidget(self.exercise_prompt_label)

        right_vertical_splitter.addWidget(lesson_content_widget)

        # Inner vertical splitter for Code Editor and AI Interaction
        code_ai_splitter = QSplitter(Qt.Vertical)

        # 2. Your Code Section (no changes needed here)
        code_editor_widget = QWidget()
        code_editor_layout = QVBoxLayout(code_editor_widget)
        code_editor_layout.setContentsMargins(5,5,5,5)
        your_code_label = QLabel("Your Code:")
        your_code_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        code_editor_layout.addWidget(your_code_label)
        self.code_editor = CodeEditor()
        code_editor_layout.addWidget(self.code_editor)
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Code")
        self.run_button.clicked.connect(self.run_code)
        button_layout.addWidget(self.run_button)
        self.check_button = QPushButton("Check Answer")
        self.check_button.clicked.connect(self.check_answer)
        button_layout.addWidget(self.check_button)
        code_editor_layout.addLayout(button_layout)
        code_ai_splitter.addWidget(code_editor_widget)

        # 3. AI Interaction Section (no changes needed here)
        ai_interaction_widget = QWidget()
        ai_interaction_layout = QVBoxLayout(ai_interaction_widget)
        ai_interaction_layout.setContentsMargins(5,5,5,5)
        ai_interaction_label = QLabel("Ask the AI:")
        ai_interaction_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        ai_interaction_layout.addWidget(ai_interaction_label)
        self.ai_input_field = QLineEdit()
        self.ai_input_field.setPlaceholderText("Type your question here (e.g., 'What is a variable?', 'How does def work?')")
        ai_interaction_layout.addWidget(self.ai_input_field)
        self.ask_ai_button = QPushButton("Ask AI for General Help")
        self.ask_ai_button.clicked.connect(self.ask_ai_question)
        ai_interaction_layout.addWidget(self.ask_ai_button)
        code_ai_splitter.addWidget(ai_interaction_widget)

        right_vertical_splitter.addWidget(code_ai_splitter)
        main_splitter.addWidget(right_vertical_splitter)

        main_splitter.setSizes([300, 900])
        right_vertical_splitter.setSizes([300, 500])
        code_ai_splitter.setSizes([400, 150])

    def ask_ai_question(self):
        user_query = self.ai_input_field.text().strip()
        if not user_query:
            self.output_text_edit.append("\nAI Interaction: Please type a question before clicking 'Ask AI'.")
            return

        self.output_text_edit.append(f"\n--- You asked the AI ---\n{user_query}")
        self.output_text_edit.append("\nAI Interaction: Getting response from AI...")
        QApplication.processEvents() # Update UI immediately

        ai_response = self.agent.ask_general_question(user_query)
        self.output_text_edit.append(f"\n{ai_response}")
        self.ai_input_field.clear() # Clear the input field after sending

    def lesson_selected(self, index):
        self.current_lesson_index = index
        self.current_exercise_index = 0
        self.load_lesson_content(index)
        
        # Ensure lesson_id is derived robustly
        if 0 <= self.current_lesson_index < len(self.lessons_data):
            lesson_entry = self.lessons_data[self.current_lesson_index]
            lesson_id = lesson_entry.get('id', lesson_entry.get('title', f"lesson_{self.current_lesson_index}").replace(" ", "_").lower())
            
            for exercise_idx in range(len(lesson_entry.get('exercise_checks', []))):
                key = f"{lesson_id}-{exercise_idx}"
                self.exercise_attempts[key] = 0
        else:
            self.output_text_edit.append("Error: Selected lesson index is out of bounds. Cannot initialize exercise attempts.")


    def load_lesson_content(self, index):
        if not (0 <= index < len(self.lesson_topics)):
            self.output_text_edit.setText("No lesson topic selected or index out of bounds.")
            self.current_exercise_label.hide()
            self.exercise_prompt_label.hide()
            self.lesson_content_display.setMarkdown("")
            self.base_lesson_content = "No lesson selected or index out of bounds."
            self.current_exercises_data = []
            return

        topic = self.lesson_topics[index]
        lesson_id = self.lessons_data[index].get('id', topic.replace(" ", "_").lower()) 
        
        ai_generated_success = False
        generated_data = None 

        # --- Try to load from local file first if it exists and looks valid ---
        found_lesson_in_json = None
        for lesson_entry in self.lessons_data:
            if lesson_entry.get('title') == topic:
                found_lesson_in_json = lesson_entry
                break
        
        if found_lesson_in_json and found_lesson_in_json.get('filename'):
            md_file_path = os.path.join(os.path.dirname(__file__), "lessons", found_lesson_in_json['filename'])
            if os.path.exists(md_file_path):
                try:
                    with open(md_file_path, 'r', encoding='utf-8') as f:
                        disk_content = f.read()
                        if "Error: Could not generate lesson content" not in disk_content and disk_content.strip():
                            self.base_lesson_content = disk_content
                            self.current_exercises_data = found_lesson_in_json.get('exercise_checks', [])
                            self.output_text_edit.append(f"\nLoaded '{topic}' from local file '{found_lesson_in_json['filename']}'.")
                            ai_generated_success = True # Treat as success because we have valid content
                        else:
                            self.output_text_edit.append(f"\nWarning: Local file '{md_file_path}' for '{topic}' contained previous error message or was empty. Attempting AI generation...")
                except Exception as e:
                    self.output_text_edit.append(f"\nError reading local file '{md_file_path}': {e}. Attempting AI generation...")
            else:
                self.output_text_edit.append(f"\nInfo: Local file '{md_file_path}' for '{topic}' not found. Attempting AI generation...")
        else:
            self.output_text_edit.append(f"\nInfo: No local metadata or filename found for '{topic}'. Attempting AI generation...")


        # --- Only try AI if local load failed or was problematic ---
        if not ai_generated_success and self.agent.api_status == "READY":
            self.output_text_edit.append(f"\nAI: Attempting to generate lesson content for '{topic}'...")
            QApplication.processEvents() # Update UI immediately

            generated_data = self.agent.generate_lesson_content(topic)

            if generated_data and generated_data.get('lesson_content_markdown') and \
               "Error: Could not generate lesson content" not in generated_data.get('lesson_content_markdown'):
                
                self.base_lesson_content = generated_data.get('lesson_content_markdown')
                self.current_exercises_data = generated_data.get('exercises', [])
                self.output_text_edit.append(f"\nAI: Lesson '{topic}' generated successfully.")
                ai_generated_success = True
                
                # --- Save the AI-generated content locally ---
                self.save_lesson_to_file(
                    lesson_index=index,
                    lesson_title=topic,
                    lesson_markdown=self.base_lesson_content,
                    exercises=self.current_exercises_data
                )
            else:
                self.output_text_edit.append(f"\nAI: Generation failed for '{topic}'. {generated_data.get('lesson_content_markdown', '') if generated_data else 'No data generated.'} Content remains unavailable.")
        
        # --- Final fallback if nothing worked ---
        if not ai_generated_success:
            self.base_lesson_content = "Lesson content unavailable. Please ensure your API key is correct and try regenerating."
            self.current_exercises_data = []

        # Update the UI displays
        self.update_exercise_prompt_in_display()

        # Set initial code for the first exercise if available, or clear
        if self.current_exercises_data and self.current_exercise_index < len(self.current_exercises_data):
            initial_code_for_current_exercise = self.current_exercises_data[self.current_exercise_index].get('initial_code', '')
            self.code_editor.setText(initial_code_for_current_exercise)
        else:
            self.code_editor.setText("") # Clear code editor if no exercises or at end

    def update_exercise_prompt_in_display(self):
        # 1. Always set the base lesson content first to the main display.
        if hasattr(self, 'lesson_content_display'):
            self.lesson_content_display.setMarkdown(self.base_lesson_content)
            self.lesson_content_display.verticalScrollBar().setValue(0)
        else:
            print("Error: lesson_content_display not found during update_exercise_prompt_in_display")
            self.current_exercise_label.setText("Error: UI component missing.")
            self.exercise_prompt_label.setText("")
            self.current_exercise_label.show()
            self.exercise_prompt_label.show()
            return # Exit early if critical component missing

        # 2. Then, handle the exercise specific labels.
        if 0 <= self.current_lesson_index < len(self.lessons_data):
            current_lesson = self.lessons_data[self.current_lesson_index]
            exercises = current_lesson.get('exercise_checks', [])

            if 0 <= self.current_exercise_index < len(exercises):
                prompt_text = exercises[self.current_exercise_index]['prompt']
                self.current_exercise_label.setText("Current Exercise:")
                self.exercise_prompt_label.setText(prompt_text)
                self.current_exercise_label.show()
                self.exercise_prompt_label.show()
            else:
                self.current_exercise_label.setText("Lesson Exercises Completed!")
                self.exercise_prompt_label.setText("You've finished all exercises for this lesson. Great job!")
                self.current_exercise_label.show()
                self.exercise_prompt_label.show()
        else:
            self.current_exercise_label.setText("")
            self.exercise_prompt_label.setText("")
            self.current_exercise_label.hide()
            self.exercise_prompt_label.hide()

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
                    self.code_editor.setText(exercises[self.current_exercise_index].get('initial_code', ''))
                else:
                    self.code_editor.setText(lesson.get('initial_code', '')) # Should clear if no more exercises

            if self.current_exercise_index >= len(exercises):
                self.output_text_edit.append("Congratulations! You've completed all exercises for this lesson.")
        else:
            self.output_text_edit.append("Incorrect. Please try again.")

            feedback_message = self.agent.provide_feedback_on_code(
                user_code=user_code,
                problem_description=problem_description,
                expected_output=expected_output,
                previous_errors=execution_errors + "\n" + captured_output_str,
                num_attempts=num_attempts
            )
            self.output_text_edit.append(f"\n--- Agent Feedback ---\n{feedback_message}")

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