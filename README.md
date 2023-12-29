# Conversation-Summarizer

Problem Statement:

Build a summary of conversations on Twitter between Customer Support Agents(AI Agents) and User to get the whole idea of ‘what problem a specific user is facing’.

Model:

To build a model for this problem , we would be using a dataset of twitter conversations from https://huggingface.co/datasets/Salesforce/dialogstudio/viewer/TweetSumm and would be training this on Llama 2 which is an Open Source Large Language Model.

Workflow:

This code demonstrates how to fine-tune a large language model (LLM) called LaMDA 2-7B for conversation summarization using the Bits-and-Bytes quantization technique. The model is fine-tuned on a dataset of conversations between humans and AI agents, where each conversation is paired with a human-written summary of the conversation. The code performs the following steps:

Data Preprocessing:

Loads the Salesforce/dialogstudio dataset, consisting of conversations between humans and AI agents on various topics. Each conversation is accompanied by a human-written summary.
Preprocesses the data by:
Cleaning the text, removing special characters, and converting everything to lowercase.
Removing irrelevant information like original dialog IDs, new dialog IDs, dialog indexes, original dialog info, logs, and prompts.
Generating training prompts in the format "Input: [conversation] Response: [summary]".

Model creation:

Defines a function to create the LLM and tokenizer.
Uses the AutoModelForCausalLM.from_pretrained function to download the pre-trained LaMDA 2-7B model and configures it for quantization using the Bits-and-Bytes (BnB) technique.
BnB quantization helps reduce the model size and make it more efficient for inference on devices with limited resources.
Creates a tokenizer for the model to convert text into numerical representations that the model can understand.

Training:

Defines a function to train the model.
Sets up the training arguments, including:
Training batch size: Number of conversation-summary pairs processed together.
Learning rate: Controls how much the model updates its weights based on the training data.
Number of training epochs: Number of times the model iterates through the entire training dataset.
Creates a Trainer object and trains the model on the prepared dataset using the Trainer's training loop.

Inference:

Defines a function to summarize new conversations using the fine-tuned model.
Takes a conversation as input, preprocesses it similarly to the training data, and feeds it to the model.
The model generates a summary of the conversation based on its understanding of the conversation and the training data it was fine-tuned on.

Additional points:

The code uses the Lora architecture for quantization, which is a specific type of quantization that is well-suited for LLMs and helps preserve accuracy during the quantization process.
The code trains the model on a TPUv4 accelerator, which is a powerful type of hardware that can significantly speed up training, making it feasible to train large models like LaMDA 2-7B.
The code evaluates the model's performance using the BLEU score, a common metric for evaluating the quality of machine-generated summaries. It compares the generated summaries with human-written summaries and calculates a score based on how similar they are.
The code could be extended to support other summarization tasks, such as summarizing news articles or scientific papers, by fine-tuning the model on different datasets specific to those domains.











