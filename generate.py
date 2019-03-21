import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy

from models import RNN, GRU
from models import make_model as TRANSFORMER



##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN, GRU, TRANSFORMER)')
parser.add_argument('--optimizer', type=str, default='SGD_LR_SCHEDULE',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
parser.add_argument('--seq_len', type=int, default=35,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=20,
                    help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')
parser.add_argument('--save_best', action='store_true',
                    help='save the model for the best validation performance')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of hidden layers in RNN/GRU, or number of transformer blocks in TRANSFORMER')

parser.add_argument('--load_model', type=str, default='',
                    help='path to model to load')


# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

# Arguments that you may want to make use of / implement more code for
parser.add_argument('--debug', action='store_true')
parser.add_argument('--save_dir', type=str, default='',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')
parser.add_argument('--evaluate', action='store_true',
                    help="use this flag to run on the test set. Only do this \
                    ONCE for each model setting, and only after you've \
                    completed ALL hyperparameter tuning on the validation set.\
                    Note we are not requiring you to do this.")

# DO NOT CHANGE THIS (setting the random seed makes experiments deterministic,
# which helps for reproducibility)
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

# # Use the model, optimizer, and the flags passed to the script to make the
# # name for the experimental dir
# print("\n########## Setting Up Experiment ######################")
# flags = [flag.lstrip('--') for flag in sys.argv[1:]]
# experiment_path = os.path.join(args.save_dir+'_'.join([argsdict['model'],
#                                          argsdict['optimizer']]
#                                          + flags))
#
# # Increment a counter so that previous results with the same args will not
# # be overwritten. Comment out the next four lines if you only want to keep
# # the most recent results.
# i = 0
# while os.path.exists(experiment_path + "_" + str(i)):
#     i += 1
# experiment_path = experiment_path + "_" + str(i)



# # LOAD DATA
# print('Loading data from '+args.data)
# raw_data = ptb_raw_data(data_path=args.data)
# train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
# vocab_size = len(word_to_id)
# print('  vocabulary size: {}'.format(vocab_size))




if args.model == 'RNN':
    model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size,
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=10000, num_layers=args.num_layers,
                dp_keep_prob=args.dp_keep_prob)
elif args.model == 'GRU':
    model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size,
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=10000, num_layers=args.num_layers,
                dp_keep_prob=args.dp_keep_prob)


model.load_state_dict(torch.load(args.load_model, map_location='cpu'))
# model.to(device) ???
model.eval()
print(model)
model.generate(1,2,3)  ###  input, hidden, generated_seq_len)

"""
Arguments:
    - input: A mini-batch of input tokens (NOT sequences!)
                    shape: (batch_size)
    - hidden: The initial hidden states for every layer of the stacked RNN.
                    shape: (num_layers, batch_size, hidden_size)
    - generated_seq_len: The length of the sequence to generate.
                   Note that this can be different than the length used
                   for training (self.seq_len)
Returns:
    - Sampled sequences of tokens
                shape: (generated_seq_len, batch_size)
"""