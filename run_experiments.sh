# Experiments SEQ2SEQ MIT-BIH
#python train.py --dataset MIT-BIH --model Seq2Seq --path models/saved/MIT-BIH/Seq2Seq --return_sequences
# Experiments SEQ2SEQ INCART
#python train.py --dataset INCART --model Seq2Seq --path models/saved/INCART/Seq2Seq --return_sequences

# Experiments CNN MIT-BIH
python train.py --dataset MIT-BIH --model CNN --path models/saved/MIT-BIH/CNN

# Experiments CNN INCART
python train.py --dataset INCART --model CNN --path models/saved/INCART/CNN