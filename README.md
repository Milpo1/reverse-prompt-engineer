# Reverse Prompt Engineer

A Python command-line tool that reverse engineers the original prompt used to generate specific outputs from given inputs.

## Table of Contents

- [Reverse Prompt Engineer](#reverse-prompt-engineer)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
  - [Examples](#examples)
    - [Sample Files](#sample-files)
    - [Running the Tool](#running-the-tool)
  - [Adding Your Own Data](#adding-your-own-data)
    - [Prepare Samples](#prepare-samples)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

This tool takes input-output pairs (samples) and attempts to reconstruct the original prompt that was used to generate the outputs from the inputs. 

Supported LLM providers:
- OpenAI
- Anthropic
- Google Gemini

## Features
It's best to illustrate with ane example. Given this sample as input: \
\
**samples.json**
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
The Reverse Prompt Engineer provides you with a reconstructed, initial prompt that could be the source of such output: \
\


## Setup

### Prerequisites

- Python 3.7 or higher
- API keys for the LLM providers you wish to use as environment variables:
  - OPENAI_API_KEY
  - ANTHROPIC_API_KEY
  - GEMINI_API_KEY

### Installation

```bash
git clone https://github.com/Milpo1/reverse-engineer-prompt-tool.git
cd reverse-engineer-prompt-tool

pip install -r requirements.txt
```

## Usage

```bash
python reverse_engineer.py --provider PROVIDER --samples PATH_TO_SAMPLES [--test]
```

- `--provider`: The LLM provider to use (openai, gemini, or anthropic).
- `--samples`: Path to a JSON file containing samples.
- `--test`: (Optional) Test the reconstructed prompt on the samples.

## Examples

### Sample Files

### Running the Tool

```bash
python reverse_engineer.py --provider openai --samples samples.json --test
```

## Adding Your Own Data

To use your own data:

### Prepare Samples

Create a JSON file with your samples in the following format:

```json
[
  {
    "variables": {
      "variable_name1": "value1",
      "variable_name2": "value2"
    },
    "response": "Expected response text"
  }
]
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License.