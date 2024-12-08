# Reverse Prompt Engineer

A Python command-line tool that reverse engineers the original prompt used to generate specific outputs from given inputs. It leverages Large Language Models (LLMs) to iteratively refine the reconstructed prompt.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Examples](#examples)

## Overview

This tool takes input-output pairs (samples) and attempts to reconstruct the original natural language prompt that was used to generate the outputs from the inputs when given to an LLM. It iteratively improves the prompt based on generated output differences.

**Illustrative Example:**

Given the following sample in `samples.json`:

```json
[
  {
    "variables": {
      "meal_log": "Day 1\nBreakfast: Oatmeal\nLunch: Salad\nDinner: Chicken",
      "patient_name": "Alex",
      "patient_surname": "Johnson",
      "goal": "Healthy eating"
    },
    "response": "{\n  \"feedback\": \"Great job, Alex! Keep up the healthy eating habits!\"\n}"
  }
]
```
The Reverse Prompt Engineer might reconstruct an initial prompt like this:
```
Provide feedback in JSON format to {{patient_name}} about their meal log: {{meal_log}}. The goal is {{goal}}.
```

## Prerequisites

- Python 3.7 or higher

API keys for the LLM providers you wish to use, set as environment variables:

- OPENAI_API_KEY

- ANTHROPIC_API_KEY

- GEMINI_API_KEY

## Usage
```
python reverse_engineer.py --data-file <path_to_data_file> --provider <provider> --num-samples <num_samples> --iterations <iterations> --batch-size <batch_size>
```
**Command-Line Arguments**

--data-file: Path to a JSON file containing input-output samples. (Required)

--provider: The LLM provider to use (openai, gemini, or anthropic). (Required)

--num-samples: Number of samples to use for initial reverse engineering. (Default: 1)

--iterations: Number of iterations for prompt improvement. (Default: 1)

--batch-size: Number of samples to use for each training iteration. (Default: 1)

## Examples
**Sample Data**

The samples.json file in the repository contains example input-output pairs.
```json
[
    {
        "variables": {
            "meal_log": "Day 1\nBreakfast: Oatmeal\nLunch: Salad\nDinner: Chicken",
            "patient_name": "Alex",
            "patient_surname": "Johnson",
            "goal": "Healthy eating"
        },
        "response": "{\n  \"feedback\": \"Great job, Alex! Keep up the healthy eating habits!\"\n}"
    },
    {
        "variables": {
            "meal_log": "Day 1\nBreakfast: Cereal\nLunch: Pizza\nDinner: Burger",
            "patient_name": "Bob",
            "patient_surname": "Smith",
            "goal": "Weight loss"
        },
        "response": "{\n  \"feedback\": \"Bob, consider healthier options for your meals to support your weight loss goal.\"\n}"
    }
]
```
**Running the Tool with Examples**

To run the tool using the provided samples.json and OpenAI as the provider:

```bash
python reverse_engineer.py --data-file samples.json --provider openai --num-samples 2 --iterations 3 --batch-size 2
```

This command will:

Use the first 2 samples from samples.json for initial prompt reconstruction.

Run 3 iterations to improve the prompt.

Use a batch size of 2 samples in each iteration.

**Adding Your Own Data**

To use your own data, create a JSON file with samples in the following format:

```json
[
  {
    "variables": {
      "variable_name1": "value1",
      "variable_name2": "value2"
      // ... more variables
    },
    "response": "Expected response text for these variable values"
  },
  // ... more samples
]
```

Then, run the tool with the --data-file argument pointing to your JSON file.
