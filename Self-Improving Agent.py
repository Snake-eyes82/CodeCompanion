
import google.generativeai as genai
import json
import time
import re
from typing import Dict, List, Any
from datetime import datetime
import traceback
     

class SelfImprovingAgent:
    def __init__(self, api_key: str):
        """Initialize the self-improving agent with Gemini API"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        self.memory = {
            'successful_strategies': [],
            'failed_attempts': [],
            'learned_patterns': [],
            'performance_metrics': [],
            'code_improvements': []
        }

        self.capabilities = {
            'problem_solving': 0.5,
            'code_generation': 0.5,
            'learning_efficiency': 0.5,
            'error_handling': 0.5
        }

        self.iteration_count = 0
        self.improvement_history = []

    def analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze a given task and determine approach"""
        analysis_prompt = f"""
        Analyze this task and provide a structured approach:
        Task: {task}

        Please provide:
        1. Task complexity (1-10)
        2. Required skills
        3. Potential challenges
        4. Recommended approach
        5. Success criteria

        Format as JSON.
        """

        try:
            response = self.model.generate_content(analysis_prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "complexity": 5,
                    "skills": ["general problem solving"],
                    "challenges": ["undefined requirements"],
                    "approach": "iterative improvement",
                    "success_criteria": ["task completion"]
                }
        except Exception as e:
            print(f"Task analysis error: {e}")
            return {"complexity": 5, "skills": [], "challenges": [], "approach": "basic", "success_criteria": []}

    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """Attempt to solve a problem using current capabilities"""
        self.iteration_count += 1
        print(f"\n=== Iteration {self.iteration_count} ===")
        print(f"Problem: {problem}")

        task_analysis = self.analyze_task(problem)
        print(f"Task Analysis: {task_analysis}")

        solution_prompt = f"""
        Based on my previous learning and capabilities, solve this problem:
        Problem: {problem}

        My current capabilities: {self.capabilities}
        Previous successful strategies: {self.memory['successful_strategies'][-3:]}  # Last 3
        Known patterns: {self.memory['learned_patterns'][-3:]}  # Last 3

        Provide a detailed solution with:
        1. Step-by-step approach
        2. Code implementation (if applicable)
        3. Expected outcome
        4. Potential improvements
        """

        try:
            start_time = time.time()
            response = self.model.generate_content(solution_prompt)
            solve_time = time.time() - start_time

            solution = {
                'problem': problem,
                'solution': response.text,
                'solve_time': solve_time,
                'iteration': self.iteration_count,
                'task_analysis': task_analysis
            }

            quality_score = self.evaluate_solution(solution)
            solution['quality_score'] = quality_score

            self.memory['performance_metrics'].append({
                'iteration': self.iteration_count,
                'quality': quality_score,
                'time': solve_time,
                'complexity': task_analysis.get('complexity', 5)
            })

            if quality_score > 0.7:
                self.memory['successful_strategies'].append(solution)
                print(f"✅ Solution Quality: {quality_score:.2f} (Success)")
            else:
                self.memory['failed_attempts'].append(solution)
                print(f"❌ Solution Quality: {quality_score:.2f} (Needs Improvement)")

            return solution

        except Exception as e:
            print(f"Problem solving error: {e}")
            error_solution = {
                'problem': problem,
                'solution': f"Error occurred: {str(e)}",
                'solve_time': 0,
                'iteration': self.iteration_count,
                'quality_score': 0.0,
                'error': str(e)
            }
            self.memory['failed_attempts'].append(error_solution)
            return error_solution

    def evaluate_solution(self, solution: Dict[str, Any]) -> float:
        """Evaluate the quality of a solution"""
        evaluation_prompt = f"""
        Evaluate this solution on a scale of 0.0 to 1.0:

        Problem: {solution['problem']}
        Solution: {solution['solution'][:500]}...  # Truncated for evaluation

        Rate based on:
        1. Completeness (addresses all aspects)
        2. Correctness (logically sound)
        3. Clarity (well explained)
        4. Practicality (implementable)
        5. Innovation (creative approach)

        Respond with just a decimal number between 0.0 and 1.0.
        """

        try:
            response = self.model.generate_content(evaluation_prompt)
            score_match = re.search(r'(\d+\.?\d*)', response.text)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)
            return 0.5
        except:
            return 0.5

    def learn_from_experience(self):
        """Analyze past performance and improve capabilities"""
        print("\n🧠 Learning from experience...")

        if len(self.memory['performance_metrics']) < 2:
            return

        learning_prompt = f"""
        Analyze my performance and suggest improvements:

        Recent Performance Metrics: {self.memory['performance_metrics'][-5:]}
        Successful Strategies: {len(self.memory['successful_strategies'])}
        Failed Attempts: {len(self.memory['failed_attempts'])}

        Current Capabilities: {self.capabilities}

        Provide:
        1. Performance trends analysis
        2. Identified weaknesses
        3. Specific improvement suggestions
        4. New capability scores (0.0-1.0 for each capability)
        5. New patterns learned

        Format as JSON with keys: analysis, weaknesses, improvements, new_capabilities, patterns
        """

        try:
            response = self.model.generate_content(learning_prompt)

            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                learning_results = json.loads(json_match.group())

                if 'new_capabilities' in learning_results:
                    old_capabilities = self.capabilities.copy()
                    for capability, score in learning_results['new_capabilities'].items():
                        if capability in self.capabilities:
                            self.capabilities[capability] = min(max(float(score), 0.0), 1.0)

                    print(f"📈 Capability Updates:")
                    for cap, (old, new) in zip(self.capabilities.keys(),
                                             zip(old_capabilities.values(), self.capabilities.values())):
                        change = new - old
                        print(f"  {cap}: {old:.2f} → {new:.2f} ({change:+.2f})")

                if 'patterns' in learning_results:
                    self.memory['learned_patterns'].extend(learning_results['patterns'])

                self.improvement_history.append({
                    'iteration': self.iteration_count,
                    'timestamp': datetime.now().isoformat(),
                    'learning_results': learning_results,
                    'capabilities_before': old_capabilities,
                    'capabilities_after': self.capabilities.copy()
                })

                print(f"✨ Learned {len(learning_results.get('patterns', []))} new patterns")

        except Exception as e:
            print(f"Learning error: {e}")

    def generate_improved_code(self, current_code: str, improvement_goal: str) -> str:
        """Generate improved version of code"""
        improvement_prompt = f"""
        Improve this code based on the goal:

        Current Code:
        {current_code}

        Improvement Goal: {improvement_goal}
        My current capabilities: {self.capabilities}
        Learned patterns: {self.memory['learned_patterns'][-3:]}

        Provide improved code with:
        1. Enhanced functionality
        2. Better error handling
        3. Improved efficiency
        4. Clear comments explaining improvements
        """

        try:
            response = self.model.generate_content(improvement_prompt)

            improved_code = {
                'original': current_code,
                'improved': response.text,
                'goal': improvement_goal,
                'iteration': self.iteration_count
            }

            self.memory['code_improvements'].append(improved_code)
            return response.text

        except Exception as e:
            print(f"Code improvement error: {e}")
            return current_code

    def self_modify(self):
        """Attempt to improve the agent's own code"""
        print("\n🔧 Attempting self-modification...")

        current_method = """
        def solve_problem(self, problem: str) -> Dict[str, Any]:
            # Current implementation
            pass
        """

        improved_method = self.generate_improved_code(
            current_method,
            "Make problem solving more efficient and accurate"
        )

        print("Generated improved method structure")
        print("Note: Actual self-modification requires careful implementation in production")

    def run_improvement_cycle(self, problems: List[str], cycles: int = 3):
        """Run a complete improvement cycle"""
        print(f"🚀 Starting {cycles} improvement cycles with {len(problems)} problems")

        for cycle in range(cycles):
            print(f"\n{'='*50}")
            print(f"IMPROVEMENT CYCLE {cycle + 1}/{cycles}")
            print(f"{'='*50}")

            cycle_results = []
            for problem in problems:
                result = self.solve_problem(problem)
                cycle_results.append(result)
                time.sleep(1)

            self.learn_from_experience()

            if cycle < cycles - 1:
                self.self_modify()

            avg_quality = sum(r.get('quality_score', 0) for r in cycle_results) / len(cycle_results)
            print(f"\n📊 Cycle {cycle + 1} Summary:")
            print(f"  Average Solution Quality: {avg_quality:.2f}")
            print(f"  Current Capabilities: {self.capabilities}")
            print(f"  Total Patterns Learned: {len(self.memory['learned_patterns'])}")

            time.sleep(2)

    def get_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        if not self.memory['performance_metrics']:
            return "No performance data available yet."

        metrics = self.memory['performance_metrics']
        avg_quality = sum(m['quality'] for m in metrics) / len(metrics)
        avg_time = sum(m['time'] for m in metrics) / len(metrics)

        report = f"""
        📈 AGENT PERFORMANCE REPORT
        {'='*40}

        Total Iterations: {self.iteration_count}
        Average Solution Quality: {avg_quality:.3f}
        Average Solve Time: {avg_time:.2f}s

        Successful Solutions: {len(self.memory['successful_strategies'])}
        Failed Attempts: {len(self.memory['failed_attempts'])}
        Success Rate: {len(self.memory['successful_strategies']) / max(1, self.iteration_count) * 100:.1f}%

        Current Capabilities:
        {json.dumps(self.capabilities, indent=2)}

        Patterns Learned: {len(self.memory['learned_patterns'])}
        Code Improvements: {len(self.memory['code_improvements'])}
        """

        return report
     

def main():
    """Main function to demonstrate the self-improving agent"""

    API_KEY = "your-gemini-api-key-here"  # Replace with your actual Gemini API key

    if API_KEY == "your-gemini-api-key-here":
        print("⚠️  Please set your Gemini API key in the API_KEY variable")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return

    agent = SelfImprovingAgent(API_KEY)

    test_problems = [
        "Write a function to calculate the factorial of a number",
        "Create a simple text-based calculator that handles basic operations",
        "Design a system to find the shortest path between two points in a graph",
        "Implement a basic recommendation system for movies based on user preferences",
        "Create a machine learning model to predict house prices based on features"
    ]

    print("🤖 Self-Improving Agent Demo")
    print("This agent will attempt to solve problems and improve over time")

    agent.run_improvement_cycle(test_problems, cycles=3)

    print("\n" + agent.get_performance_report())

    print("\n" + "="*50)
    print("TESTING IMPROVED AGENT")
    print("="*50)

    final_problem = "Create an efficient algorithm to sort a large dataset"
    final_result = agent.solve_problem(final_problem)

    print(f"\nFinal Problem Solution Quality: {final_result.get('quality_score', 0):.2f}")
     

def setup_instructions():
    """Print setup instructions for Google Colab"""
    instructions = """
    📋 SETUP INSTRUCTIONS FOR GOOGLE COLAB:

    1. Install the Gemini API client:
       !pip install google-generativeai

    2. Get your Gemini API key:
       - Go to https://makersuite.google.com/app/apikey
       - Create a new API key
       - Copy the key

    3. Replace 'your-gemini-api-key-here' with your actual API key

    4. Run the code!

    🔧 CUSTOMIZATION OPTIONS:
    - Modify test_problems list to add your own challenges
    - Adjust improvement cycles count
    - Add new capabilities to track
    - Extend the learning mechanisms

    💡 IMPROVEMENT IDEAS:
    - Add persistent memory (save/load agent state)
    - Implement more sophisticated evaluation metrics
    - Add domain-specific problem types
    - Create visualization of improvement over time
    """
    print(instructions)

if __name__ == "__main__":
    setup_instructions()
    print("\n" + "="*60)
    main()