from nltk.util import ngrams
from nltk.lm.preprocessing import flatten, padded_everygram_pipeline
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.lm import MLE
import argparse
import os
import random
import io 
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Process input file, output path, and n value.")
    parser.add_argument('--input_file', type=str, default='None',help='Name of the input file')
    parser.add_argument('--output_file', type=str, default ='None', help='Path to save the output file')
    parser.add_argument('--n', type=int, default = 2, help='N in n-gram')
    args = parser.parse_args()
    

    return args


args = parse_args()


input_file = args.input_file

def tokenize(text):
    return [word_tokenize(sent) for sent in sent_tokenize(text)][0]

try: 
    from nltk import word_tokenize, sent_tokenize 
    word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])
except:
    import re
    from nltk.tokenize import ToktokTokenizer

    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)

    toktok = ToktokTokenizer()
    word_tokenize = word_tokenize = toktok.tokenize



if os.path.isfile(input_file):
    with io.open(input_file, encoding='utf8') as fin:
        text = fin.read()
else:
    exit()



tokenized_text = [list(map(str.lower, word_tokenize(sent))) 
                  for sent in sent_tokenize(text)]



num_tokens = sum([len(text) for text in tokenized_text])

print('Total number of tokens is', num_tokens)



n = args.n
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)


model = MLE(n)


model.fit(train_data, padded_sents)
print('Total unique vocabulary is', len(model.vocab))


detokenize = TreebankWordDetokenizer().detokenize
n_1_grams = list(flatten([ngrams(sentence, n-1) for sentence in tokenized_text]))
print('Total n-1 grams are', len(n_1_grams))
print('Each context is of size', len(n_1_grams[0]))


def generate_sent(model, num_words, n_1_grams=n_1_grams):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    
    context = list(random.choice(n_1_grams))
    content = []

    generated_text = context[:]

    for _ in range(num_words):
        next_word = model.generate(text_seed=context)
        
        if next_word == '<s>':
            continue
        if next_word == '</s>':
            break
        
        generated_text.append(next_word)

        context = context[1:] + [next_word]
    
    return ' '.join(generated_text).replace("<s>", "").replace("</s>", "").strip()



pbar = tqdm(total=num_tokens)

def generate_and_write_to_file(model, m, output_file, num_words=50):
    """
    Generates m tokens and writes to a file

    model: nltk
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """

    total_tokens = 0
    
    with open(output_file, 'w') as f:
        while total_tokens < m:
            generated_sentence = generate_sent(model, num_words)
            temp = len(tokenize(generated_sentence))
            total_tokens += temp
            pbar.update(temp)
            f.write(''.join(generated_sentence) + '\n')
            # print(f"Generated {total_tokens}/{m} tokens so far...")

    print(f"Generation complete. Saved to {output_file}.")



m = num_tokens 

generate_and_write_to_file(model, m, args.output_file)
