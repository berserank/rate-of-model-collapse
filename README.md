# Rate of Model Collapse in Recursive Training

This repository contains the code for all the simulations discussed in the paper, 'Rate of Model Collapse in Recursive Training.' In this study, we analyze the rate of model collapse during recursive training in Bernoulli, Poisson, and Gaussian settings. This analysis was extended to N-gram Language Models and Gaussian Mixture Models.


## Description
For all the models except the N-gram Language Model, code for recursive training simulations can be found in `simulations.ipynb` notebook. Additionally, this project includes a Python script for training and using an n-gram language model to generate text and a shell script to automate the recursive training process. The n-gram model uses the Maximum Likelihood Estimation (MLE) method from NLTK, and the text generation process produces sentences based on the statistical structure of the input text.

---

## Usage for N-gram LMs

The Python script `ngram_sim.py` is designed to train an n-gram model and generate text. Number of samples generated will be the same as number of samples in the training dataset.

#### Required Arguments:
- `--input_file`: Path to the input text file to be processed.
- `--output_file`: Path where the generated text will be saved.
- `--n`: The value of `n` for the n-gram model (e.g., 2 for bigrams, 3 for trigrams).

#### Commands:
```bash
python ngram_sim.py --input_file data/input.txt --output_file data/output.txt --n 3
```
The `ngram_sim.sh` script automates running the `ngram_sim.py` python script multiple times with n=2 (Bigram Model). It facilitates recursive training for 10 generations and batch processing of text generation using an n-gram language model.

```bash
bash ngram_sim.sh
```

After generating text, `postprocess.py` is used to analyse the change in vocabulary size during N-gram LM recursive training. 

1. Provide a path to the original training file (`train_file`), the folder containing the generated files (`generations_folder`), and the desired output file path to store vocabulary size reduction array (`output_file`).
2. Run the script, which will generate a plot showing vocabulary reduction across generations and save the reduction ratios.

```bash
python postprocess.py
