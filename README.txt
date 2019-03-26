IFT6135 - Deep Learning Assignment 2
March 25, 2019

Lawrence Abdulnour
Alexandre Marcil
Louis-François Préville-Ratelle


Q1, Q2, Q3 : models.py


Q4 : use ptb-lm.py to train the models, with desired hyperparameters in arguments.

python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best


Q5 : trained models filename (best_params.pt) must be passed in arguments to load trained models.
(i.e. with extra argument : ---load_model="best_params.pt")


Q5.1 : avg_loss_5_1.py

python avg_loss_5_1.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 ---load_model="best_params.pt"


Q5.2 : gradients_5_2.py

python gradients_5_2.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 ---load_model="best_params.pt"


Q5.3 : generate_samples_5_3.py

python generate_samples_5_3.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 ---load_model="best_params.pt"



