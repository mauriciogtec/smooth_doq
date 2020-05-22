#!usr/bin/bash
cd data/raw/
# split --additional-suffix=.csv -d -l 512000 ../DoQ_raw_nouns_noun_obj_quantization.csv doq_raw
# split --additional-suffix=.csv -d -l 512000 ../DoQ_raw_adjs_adj_obj_quantization.csv doq_raw
split --additional-suffix=.csv -d -l 512000 ../DoQ_raw_verbs_verb_obj_quantization.csv doq_raw