# This is a sample script that trains the Gaussian mixture representations for multi-sense embeddings on deWac, a German text corpus.
mkdir modelfiles
./multift skipgram -input "data/dewac.txt" -output modelfiles/multi_dewac_e10_d300_vs2e-4_lr1e-5_margin1 -lr 1e-5 -dim 300 \
    -ws 10 -epoch 10 -minCount 5 -loss ns -bucket 2000000 \
    -minn 3 -maxn 6 -thread 62 -t 1e-5 -lrUpdateRate 100 -multi 1 -var_scale 2e-4 -margin 1