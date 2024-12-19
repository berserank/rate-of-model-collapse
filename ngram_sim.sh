mkdir tmp
python ngram_sim.py --input_file train_sampled.txt --output_file tmp/generation1.txt --n 2
python ngram_sim.py --input_file tmp/generation1.txt --output_file tmp/generation2.txt --n 2
python ngram_sim.py --input_file tmp/generation2.txt --output_file tmp/generation3.txt --n 2
python ngram_sim.py --input_file tmp/generation3.txt --output_file tmp/generation4.txt --n 2
python ngram_sim.py --input_file tmp/generation4.txt --output_file tmp/generation5.txt --n 2
python ngram_sim.py --input_file tmp/generation5.txt --output_file tmp/generation6.txt --n 2
python ngram_sim.py --input_file tmp/generation6.txt --output_file tmp/generation7.txt --n 2
python ngram_sim.py --input_file tmp/generation7.txt --output_file tmp/generation8.txt --n 2
python ngram_sim.py --input_file tmp/generation8.txt --output_file tmp/generation9.txt --n 2
python ngram_sim.py --input_file tmp/generation9.txt --output_file tmp/generation10.txt --n 2