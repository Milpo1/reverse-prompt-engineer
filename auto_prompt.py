# %%
import argparse
import json
import os
import random
import re
import sys
from typing import List, Dict

# Import necessary LLM provider libraries
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

# try:
import google.generativeai as genai
# except ImportError:
#     genai = None


def taggify(tag: str, content: str) -> str:
    return f"<{tag}>\n{content}\n</{tag}>\n"


def build_reverse_engineer_prompt(samples: List[Dict], requirements: str = None) -> str:
    """
    Builds the reverse engineering prompt using the provided samples.

    Args:
        samples (List[Dict]): A list of dictionaries containing 'variables' and 'response'.
        
    Returns:
        str: The constructed reverse engineering prompt.
    """
    prompt = f"""
Your task is to reverse engineer the original prompt that was used to generate specific outputs from given inputs.

---

**Instructions:**

1. **Understand the Goal:**

    - You are provided with multiple examples. Each example includes <input_data> and the corresponding <output>.
    - The <input_data> consists of variables wrapped in tags named after the variable (e.g., <patient_name>, <meal_log>).
    - The original prompt is a set of instructions written in natural language, using special syntax to inject these variables.

2. **Analyze the Examples:**

    - Examine how the variables in <input_data> are used to produce the <output>.
    - Identify patterns, recurring phrases, and structures in the outputs that hint at the wording of the original prompt.

4. **Reconstruct the Original Prompt:**

    - **Write the prompt in natural language**, as it would be given to an assistant before variable injection.
    - Use the following syntax for variables: {{variable_name}}
    - Use ONLY the variables that are provided in <input_data>.
    - Ensure the prompt includes all necessary instructions and context to generate the outputs from the inputs.

5. **Provide the Reconstructed Prompt:**

    - Enclose your reconstructed prompt within <reverse_engineered_prompt> tags.
    - Do not include the outputs or code templates; focus on the natural language prompt.
    - Do not separate Handlerbars from natural language - no additional markdown formatting is required.
---

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


def call_openai(prompt):
    openai.api_key = os.getenv('OPENAI_API_KEY')  # Get OpenAI API key from env variable
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model='gpt-4o-2024-11-20',
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0,
        stop=None
    )
    return response.choices[0].message.content

def call_gemini(prompt):
    gemini_api_key = os.getenv('GEMINI_API_KEY')  # Get Gemini API key from env variable
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text

def call_anthropic(prompt):
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')  # Get Anthropic API key from env variable
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0,
        system="You are a helpful assistant. Respond concisely.",
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def call_llm_provider(provider: str, prompt: str) -> str:
    """
    Calls the specified LLM provider with the given prompt.

    Args:
        provider (str): The provider to use ('openai', 'gemini', or 'anthropic').
        prompt (str): The prompt to send to the API.

    Returns:
        str: The response content from the API.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider == "openai":
        return call_openai(prompt)
    elif provider == "gemini":
        return call_gemini(prompt)
    elif provider == "anthropic":
        return call_anthropic(prompt)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def extract_reconstructed_prompt(response_text: str) -> str:
    """
    Extracts the content within <reverse_engineered_prompt> tags.

    Args:
        response_text (str): The full response text from the LLM.

    Returns:
        str: The extracted prompt content.
    """
    pattern = r"<reverse_engineered_prompt>(.*?)<\/reverse_engineered_prompt>"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""


def inject_variables_and_test(samples: List[Dict], reconstructed_prompt: str, provider: str):
    """
    Injects variables into the reconstructed prompt and tests it on the given samples.

    Args:
        samples (List[Dict]): The list of samples to test.
        reconstructed_prompt (str): The reconstructed prompt to inject variables into.
        provider (str): The LLM provider to use.

    Returns:
        None
    """
    for idx, sample in enumerate(samples):
        # Inject variables into the reconstructed prompt using Handlebars syntax
        prompt_with_variables = reconstructed_prompt
        for var_name, var_content in sample['variables'].items():
            placeholder = f"{{{{{var_name}}}}}"
            prompt_with_variables = prompt_with_variables.replace(placeholder, var_content)

        # Call the LLM with the new prompt
        try:
            output = call_llm_provider(provider, prompt_with_variables)
            print(f"\nTest Output for Sample {idx+1}:")
            print(output.strip())
            print("\nExpected Response:")
            print(sample['response'])
        except Exception as e:
            print(f"Error testing sample {idx+1}: {e}")


def load_samples_from_file(file_path: str) -> List[Dict]:
    """
    Loads samples from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing samples.

    Returns:
        List[Dict]: A list of samples.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        samples = json.load(file)
    return samples


def main():
    parser = argparse.ArgumentParser(description='Reverse engineer a prompt using LLM providers.')
    parser.add_argument('--provider', type=str, choices=['openai', 'gemini', 'anthropic'], required=True,
                        help='The LLM provider to use.')
    parser.add_argument('--samples', type=str, required=True,
                        help='Path to the JSON file containing samples.')
    parser.add_argument('--requirements', type=str, required=False,
                        help='Path to the text file containing requirements.')
    parser.add_argument('--test', action='store_true',
                        help='Test the reconstructed prompt on the samples.')
    args = parser.parse_args()

    # Load samples and requirements
    try:
        samples = load_samples_from_file(args.samples)
        requirements = None
        if args.requirements: 
            with open(args.requirements, 'r', encoding='utf-8') as file:
                requirements = file.read()
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    # Build the Reverse Engineer Prompt
    reverse_engineer_prompt = build_reverse_engineer_prompt(samples, requirements)

    # Call the selected provider
    try:
        reconstructed_prompt_response = call_llm_provider(args.provider, reverse_engineer_prompt)
        # Extract the reconstructed prompt from the response
        reconstructed_prompt = extract_reconstructed_prompt(reconstructed_prompt_response)
        if not reconstructed_prompt:
            print("Could not extract reconstructed prompt from the response.")
            return
        print("Reconstructed Prompt:")
        print(reconstructed_prompt)
    except Exception as e:
        print(f"Error during prompt reconstruction: {e}")
        return

    # Optionally test the reconstructed prompt on the data samples
    if args.test:
        inject_variables_and_test(samples, reconstructed_prompt, args.provider)


if __name__ == "__main__":
    main()

# %%
