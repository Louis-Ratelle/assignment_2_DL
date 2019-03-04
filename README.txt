{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf200
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\csgray\c0;}
\margl1440\margr1440\vieww12760\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 J\'92ai fait uniquement la question 1. Mais vous devriez regarder mon code avec m\'e9fiance. \'c7a marche, mais je ne suis pas certain, surtout qu\'92une \'e9poch prend 23 minutes (sur mon ordi avec cpu).\
\
Dans un des fichiers, il est \'e9crit que nous devrions avoir (Isaac m\'92a envoy\'e9 ces valeurs, je n\'92ai pas v\'e9rifi\'e9):\
\
 RNN: train:  120  val: 157\
 GRU: train:   65  val: 104\
 TRANSFORMER:  train:  77  val: 152\
\
Mais pour RNN (probl\'e8me 1), j\'92obtiens seulement 390 avec la commande suivante (apr\'e8s 25 epochs et environ 11 heures):\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f1\fs22 \cf2 \CocoaLigature0 \
\'93 python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=16  --seq_len=35 --hidden_size=256 --num_layers=2 --dp_keep_prob=0.35 \'94\
\
\
Je n\'92ai mis aucun dropout (pour RNN Q1, je ne vois pas \'e0 quoi sert dropout.\
\
Le principal probl\'e8me avec Pytorch ici, c\'92est de bien comprendre comment fonctionne Adagrad. En d\'92autres termes, quoi utiliser pour comme blocs (torch.tensor, nn.Linear, \'85?) pour que Pytorch puisse bien faire le backprop automatiquement?\
\
\
\
\
}