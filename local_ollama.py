import subprocess
import importlib.util
import transformers
import torch
import sys
import re
from transformers import AutoTokenizer

model = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Print the device being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

user_content = input("Enter Prompt: ")

message = [
    {
    "role": "system", 
        "content": (
            "You are an assistant designed to provide complete, self-contained Python scripts in response to user prompts. You must always return Python code that is ready to execute without requiring modifications. Please follow these guidelines:"
            "1. Only output the Python code itself — no Markdown formatting, comments, explanations, or additional text."
            "2. Your output must start with any necessary import statements, followed by all required function definitions and executable code."
            "3. Ensure that the script is entirely standalone, meaning all the imports and any required data or values should be included or defined in the script."
            "4. The script should include a `main()` function or an appropriate block to run the code if executed as a standalone script."
            "5. Do not include any Markdown markers"
            "6. Prompts the user for required parameters."
            "7. Think through this carefully, step by step."
        )
    },
    {
        "role": "user", 
        "content": user_content,
    }
]

sequences = pipeline(
    message,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    truncation = True,
    max_length=1000,
)
    
def extract_python_code(sequences):
    for item in sequences[0]['generated_text']:
        if item['role'] == 'assistant':
            code_block = item['content']
            # Remove markdown markers if present
            if code_block.startswith("```python"):
                code_block = code_block[len("```python\n"):-len("\n```")]
            return code_block

def handle_imports_and_execute(code):
    # Regular expression to match import statements
    import_statements = re.findall(r'^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)', code, re.MULTILINE)
    imports = set(import_statements)

    for package in imports:
        # Check if package is already installed
        if not importlib.util.find_spec(package):
            try:
                print(f"Installing missing package: {package}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as e:
                print(f"Error installing package {package}: {e}")

    lines = code.splitlines()

    # Separate the function definitions and imports from the execution part
    definitions = []
    execution = []
    in_definitions = True

    for line in lines:
        # We assume the main execution part starts with `if __name__ == "__main__":`
        if line.startswith("if __name__ == \"__main__\":"):
            in_definitions = False
        
        if in_definitions:
            definitions.append(line)
        else:
            execution.append(line)
    
    # Join the separated parts back into two scripts
    definitions_script = "\n".join(definitions)
    execution_script = "\n".join(execution)

    # Execute the function definitions first
    exec(definitions_script, globals())

    # Then execute the main part
    exec(execution_script, globals())

# User confirmation step
def prompt_yes_no(question):
    while True:
        response = input(f"{question} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

# Call the function and print the extracted code
extracted_code = extract_python_code(sequences)

print(extracted_code)
if prompt_yes_no("Do you want to execute the generated code?"):
    handle_imports_and_execute(extracted_code)
    print("Code executed.")
else:
    print("Execution aborted by the user.")