python ngram_sim.py --input_file ngram_sim/bigram/train_sampled.txt --output_file ngram_sim/bigram/generation1.txt --n 2
python ngram_sim.py --input_file ngram_sim/bigram/generation1.txt --output_file ngram_sim/bigram/generation2.txt --n 2
python ngram_sim.py --input_file ngram_sim/bigram/generation2.txt --output_file ngram_sim/bigram/generation3.txt --n 2
python ngram_sim.py --input_file ngram_sim/bigram/generation3.txt --output_file ngram_sim/bigram/generation4.txt --n 2
python ngram_sim.py --input_file ngram_sim/bigram/generation4.txt --output_file ngram_sim/bigram/generation5.txt --n 2
python ngram_sim.py --input_file ngram_sim/bigram/generation5.txt --output_file ngram_sim/bigram/generation6.txt --n 2
python ngram_sim.py --input_file ngram_sim/bigram/generation6.txt --output_file ngram_sim/bigram/generation7.txt --n 2
python ngram_sim.py --input_file ngram_sim/bigram/generation7.txt --output_file ngram_sim/bigram/generation8.txt --n 2
python ngram_sim.py --input_file ngram_sim/bigram/generation8.txt --output_file ngram_sim/bigram/generation9.txt --n 2
python ngram_sim.py --input_file ngram_sim/bigram/generation9.txt --output_file ngram_sim/bigram/generation10.txt --n 2

