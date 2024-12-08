import os
import re
import json
import random
import argparse
import openai
import google.generativeai as genai
import anthropic
import time
import threading
from functools import wraps
import logging

# --- Constants ---
OPENAI_MODEL = 'gpt-4o-2024-11-20'
GEMINI_MODEL = "gemini-1.5-flash"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='application.log',
    filemode='a'
)

def get_function_logger(function_name):
    """Retrieves a logger for the given function name."""
    return logging.getLogger(function_name)

def log_function_calls(func):
    """Decorator to log function calls with arguments and return values."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_function_logger(func.__name__)
        logger.info(f"Called with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logger.info(f"Returned: {result}")
        return result
    return wrapper

# --- Utility Functions ---
def taggify(tag, content):
    """Wraps content in XML-like tags."""
    return f"<{tag}>\n{content}\n</{tag}>"

def rate_limit(max_calls, period):
    """
    Decorator to limit the number of times a function can be called within a time period.

    :param max_calls: Maximum number of allowed calls within the period.
    :param period: Time period in seconds.
    """
    lock = threading.Lock()
    calls = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal calls
            with lock:
                current_time = time.time()
                calls = [call for call in calls if call > current_time - period]
                if len(calls) >= max_calls:
                    wait_time = calls[0] + period - current_time
                    if wait_time > 0:
                        print(f"Rate limit exceeded. Sleeping for {wait_time:.2f} seconds.")
                        time.sleep(wait_time)
                    current_time = time.time()
                    calls = [call for call in calls if call > current_time - period]
                calls.append(time.time())
            return func(*args, **kwargs)
        return wrapper
    return decorator

# --- Prompt Building Functions ---
def build_reverse_engineer_prompt(samples):
    """
    Builds the reverse engineering prompt based on the provided samples.
    """
    prompt = f"""
Your task is to reverse engineer the original prompt that was used to generate specific outputs from given inputs.

---

**Instructions:**

1. **Understand the Goal:**
    - You are provided with multiple examples. Each example includes <input_data> and the corresponding <output>.
    - The <input_data> consists of variables wrapped in tags named after the variable (e.g., <patient_name>, <meal_log>).
    - The original prompt is a set of instructions written in natural language, using special templating syntax to inject data into the prompt.

2. **Analyze the Examples:**
    - Examine how the variables in <input_data> are used to produce the <output>.
    - Identify patterns, recurring phrases, and structures in the outputs that hint at the wording of the original prompt.

3. **Reconstruct the Original Prompt:**
    - **Write the prompt in natural language**, with special syntax used to inject data into the prompt. Remember that the LLM model will receive the prompt after data injection.
    - Use the following special syntax for data injection: {{variable_name}} (this will be replaced by the variable content)
    - Only such simple templating syntax is allowed, nothing more.
    - The reconstructed prompt shouldn't instruct to put placeholder values, unless necessary.
    - Ensure the prompt includes all necessary instructions and context to generate the outputs from the inputs.
    - Remember, the LLM model won't see the labels of the templated data, but its injected content instead.
    - Image what the prompt would look like to the LLM after injecting the data and ensure propose formating.

4. **Think out loud step by step:**
    - Within <reasoning> tags, provide you logic and thoughts on creating the prompt. Discuss the requirements and what would be important or challenging to achieve requested output. Think what actions the reverse engineered prompt took to achieve this.

5. **Provide the Reconstructed Prompt:**
    - Enclose your reconstructed prompt within <reverse_engineered_prompt> tags.
    - Do not include the outputs or code templates; focus on the natural language prompt.
    - Do not separate Handlebars from natural language - no additional markdown formatting is required.

---
**Examples:**
"""
    for sample in samples:
        input_data = ''.join(
            taggify(var_name, var_content)
            for var_name, var_content in sample['variables'].items()
        )
        input_data = taggify("input_data", input_data)
        output = taggify("output", sample['response'])
        example_output = taggify("example_output", input_data + output)
        prompt += example_output

    return prompt

def generate_differences_prompt(batch, generated_outputs):
    """
    Generates a prompt to identify differences and suggest improvements between expected and generated outputs.
    """
    prompt = """
Below are multiple samples of the provided variables, expected outputs, and generated outputs for a batch of data.

Your tasks:
1. Identify the key differences between the generated outputs and the expected outputs for all given samples.
2. Consider the variables that were used to produce the generated outputs. Data can be injected into the prompt using special syntax: {{variable_name}}.
3. Summarize what is missing or inconsistent in the generated outputs compared to the expected outputs, taking into account the variables and their intended usage.
4. Focus on how the reconstructed prompt could be modified to produce the desired format, structure, tone, content, and correct utilization of the variables.
5. Focus on the specific format the expected output is in.
6. Output ONLY plain natural language with minimal formatting (lists are allowed). NO CODE

Format your answer as a concise description of differences and needed improvements.
---
"""
    for i, (sample, expected, generated) in enumerate(zip(batch, [s['response'] for s in batch], generated_outputs), 1):
        vars_text = "\n".join([f"{k}: {v}" for k, v in sample['variables'].items()])
        prompt += f"\nSample {i}:\n"
        prompt += "Variables:\n" + vars_text + "\n\n"
        prompt += "Expected Output:\n" + expected.strip() + "\n\n"
        prompt += "Generated Output:\n" + generated.strip() + "\n"
        prompt += "----------------------------------------\n"

    return prompt

@log_function_calls
def improve_prompt(batch, current_prompt, differences, generated_outputs, provider="openai"):
    """
    Improves the current prompt based on the described differences.
    """
    prompt = f"""
The current prompt does not fully produce the expected results. Below are summarized differences and needed improvements:

{differences}

You have access to variables and can use Handlebars-like syntax for them (e.g., {{variable_name}}) and conditional blocks ({{#if condition}}...{{/if}}).

Please revise the previously reconstructed prompt so that the assistant's output aligns more closely with the expected results.
This includes correctly handling variables, ensuring the structure and formatting aligns with the desired output,
and maintaining the tone and content required.

1. **Reconstruct the Original Prompt:**
    - **Write the prompt in natural language**, with special syntax used to inject data into the prompt. Remember that the LLM model will receive the prompt after data injection.
    - Use the following special syntax for data injection: {{variable_name}}
    - Only such simple templating syntax is allowed, nothing more.
    - The reconstructed prompt shouldn't instruct to put placeholder values, unless necessary.
    - Ensure the prompt includes all necessary instructions and context to generate the outputs from the inputs.
    - Remember, the LLM model won't see the labels of the templated data, but its injected content instead.

2. **Think out loud step by step:**
    - Within <reasoning> tags, provide you logic and thoughts on creating the prompt. Discuss the requirements and what would be important or challenging to achieve requested output. Think what actions the reverse engineered prompt took to achieve this.

3. **Provide the Reconstructed Prompt:**
    - Enclose your reconstructed prompt within <reverse_engineered_prompt> tags.
    - Do not include the outputs or code templates; focus on the natural language prompt.
    - Do not separate Handlebars from natural language - no additional markdown formatting is required.

Current prompt:

<reverse_engineered_prompt>
{current_prompt}
</reverse_engineered_prompt>

---
**Sample Variable data and expected responses:**
"""
    for i, (sample, expected, generated) in enumerate(zip(batch, [s['response'] for s in batch], generated_outputs), 1):
        vars_text = "\n".join([f"{k}: {v}" for k, v in sample['variables'].items()])
        prompt += f"\nSample {i}:\n"
        prompt += "Variables:\n" + vars_text + "\n\n"
        prompt += "Expected Output:\n" + expected.strip() + "\n\n"
        prompt += "Generated Output:\n" + generated.strip() + "\n"
        prompt += "----------------------------------------\n"

    response = call_llm_provider(provider, prompt)
    improved_prompt = extract_reconstructed_prompt(response)
    return improved_prompt if improved_prompt else current_prompt

# --- LLM API Call Functions ---
def call_openai(prompt):
    """Calls the OpenAI API with the given prompt."""
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0,
        stop=None
    )
    return response.choices[0].message.content

@rate_limit(max_calls=15, period=60)
def call_gemini(prompt):
    """Calls the Gemini API with the given prompt."""
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text

def call_anthropic(prompt):
    """Calls the Anthropic API with the given prompt."""
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1000,
        temperature=0,
        system="You are a helpful assistant. Respond concisely.",
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def call_llm_provider(provider, prompt):
    """Calls the specified LLM provider with the given prompt."""
    if provider == "openai":
        return call_openai(prompt)
    elif provider == "gemini":
        return call_gemini(prompt)
    elif provider == "anthropic":
        return call_anthropic(prompt)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# --- Helper Functions for Prompt Processing ---
def extract_reconstructed_prompt(response_text):
    """Extracts the reconstructed prompt from the LLM's response."""
    pattern = r"<reverse_engineered_prompt>(.*?)<\/reverse_engineered_prompt>"
    match = re.search(pattern, response_text, re.DOTALL)
    return match.group(1).strip() if match else None

def inject_variables(reconstructed_prompt, variables):
    """Injects variables into the reconstructed prompt."""
    prompt_with_vars = reconstructed_prompt
    for var_name, var_content in variables.items():
        placeholder = f"{{{{{var_name}}}}}"
        prompt_with_vars = prompt_with_vars.replace(placeholder, str(var_content))
    return prompt_with_vars

# --- Main Function ---
def main(args):
    """Main function to run the reverse engineering and prompt improvement process."""
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = random.sample(data, min(args.num_samples, len(data)))

    print("Reverse engineering the prompt...")
    reverse_engineer_prompt = build_reverse_engineer_prompt(samples)
    reconstructed_prompt_response = call_llm_provider(args.provider, reverse_engineer_prompt)
    reconstructed_prompt = extract_reconstructed_prompt(reconstructed_prompt_response)

    if not reconstructed_prompt:
        print("Failed to extract reconstructed prompt.")
        return

    print(f"Reverse engineered prompt: {reconstructed_prompt.replace('\n', '')}")
    current_prompt = reconstructed_prompt

    for iteration in range(args.iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        batch_size = min(args.batch_size, len(data))
        batch = random.sample(data, batch_size)
        print(f"Processing a batch of {batch_size} samples.")

        generated_outputs = []
        for idx, sample in enumerate(batch):
            print(f"\nGenerating Response for Sample {idx + 1}...")
            test_prompt = inject_variables(current_prompt, sample['variables'])
            gen_out = call_llm_provider(args.provider, test_prompt)
            generated_outputs.append(gen_out)
            print(f"  Response {idx + 1} generated for sample: {sample['response'][:30].replace('\n', '')}---")

        differences_prompt = generate_differences_prompt(batch, generated_outputs)
        differences = call_llm_provider(args.provider, differences_prompt).strip()

        if differences and not differences.lower().startswith("no differences"):
            current_prompt = improve_prompt(batch, current_prompt, differences, generated_outputs, provider=args.provider)
            print("Prompt improved based on the current batch.")
        else:
            print("No improvements made for the current batch.")

    print("\nFinal Reconstructed Prompt:")
    print(current_prompt)

# --- Argument Parsing and Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse Engineer and Iteratively Improve an LLM Prompt")
    parser.add_argument("--data-file", type=str, required=True, help="Path to the JSON file containing variable-response pairs.")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini", "anthropic"], help="LLM provider to use.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to use for initial reverse engineering.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations for prompt improvement.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples to use for each training iteration.")
    args = parser.parse_args()
    main(args)