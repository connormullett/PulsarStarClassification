#!/usr/bin/env python

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys

from torch.autograd import Variable
from sklearn import preprocessing


if len(sys.argv) < 2:
  print('usage ./main {train/test} or')
  print('      ./main test positives')
  print('to only test positive values')
  exit()

arg = sys.argv[1]
if arg == 'train':
  print('Training model ...')
  is_training = True
elif arg == 'test':
  print('testing model ...')
  is_training = False
  if len(sys.argv) > 2 and sys.argv[2] == 'positives':
    print('using positive values only for evaluation')
    use_positives = True
  else:
    use_positives = False
else:
  print('use train or test as arg')
  exit()


# load data
df = pd.read_csv('./pulsar_stars.csv')

# split data
# create train/test dataframes 70/30 split
train_df = df.iloc[:12528]
test_df = df.iloc[12528:]

# create input data (x) with labels (y)
train_x = train_df.drop('target_class', 1)
train_y = train_df['target_class'].to_list()

# create test input (x) with labels for accuracy (y)
test_x = test_df.drop('target_class', 1)
test_y = test_df['target_class'].to_list()

# convert df values to float
train_x = train_x.values.astype(float)
test_x = test_x.values.astype(float)

# reshape data to -1, 1 as float
min_max_scaler = preprocessing.MinMaxScaler()

train_x = min_max_scaler.fit_transform(train_x)
train_x = pd.DataFrame(train_x)

test_x = min_max_scaler.fit_transform(test_x)
test_x = pd.DataFrame(test_x)

# Declare the model
class LogisticRegression(nn.Module):

  def __init__(self, input_dim, output_dim):
    super(LogisticRegression, self).__init__()
    self.linear = torch.nn.Linear(input_dim, output_dim)

  def forward(self, x):
    out = torch.sigmoid(self.linear(x))
    return out

if is_training:

  # estabilish params for model, loss, and optimizer
  input_dim = train_x.shape[1]
  output_dim = 1

  criterion = nn.BCELoss(reduction='mean')
  learning_rate = 0.01

  model = LogisticRegression(input_dim, output_dim)
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  [w, b] = model.parameters()

  target_tensors = torch.tensor(train_y, dtype=torch.float)

  # train
  loss_list = []
  num_iters = train_x.shape[0]
  epochs = 2
  for epoch in range(1,epochs+1):
    for iter in range(num_iters):
      # create tensor of df row for this iter
      input_data = train_x.iloc[iter]
      input_tensor = torch.tensor(input_data)

      # create the target
      target_value = target_tensors[iter].item()
      target = torch.tensor([target_value])

      # set model for training
      model.train()

      # zero out the gradient and pass data
      optimizer.zero_grad()
      output = model(input_tensor)

      # get the loss
      loss = criterion(output, target)

      # calculate derivative by stepping backwards
      loss.backward()
      loss_list.append(loss.data)

      # adjust
      optimizer.step()

      if(iter % 50 == 0):
        print('epoch {}, iter {}, loss {}'.format(epoch, iter, loss.data))


  torch.save(model.state_dict(), './model.pt')

  # visualize training as loss
  plt.plot(range(num_iters * epochs), loss_list)
  plt.xlabel('Num Iterations')
  plt.ylabel('loss')
  plt.savefig('./loss.png')



if not is_training:
  # declare hyperparameters
  input_dim = test_x.shape[1]
  output_dim = 1

  # load model
  model = LogisticRegression(input_dim, output_dim)
  state_dict = torch.load('./model.pt')
  model.load_state_dict(state_dict)

  # create tensor for test set
  eval_target_tensors = torch.tensor(test_y, dtype=torch.float)

  # define values for tracking accuracy
  correct = 0
  total = 0

  # evaluate over test set for accuracy
  for i in range(test_x.shape[0]):
    # create the inputs for evaluation
    input_data = test_x.iloc[i]
    input_tensor = torch.tensor(input_data)

    # create evaluation targets
    target_value = eval_target_tensors[i].item()

    # test accuracy of JUST positive candidates
    # (it is a pulsar)
    if use_positives:
      if not int(target_value) == 1:
        continue
    target = torch.tensor([target_value])

    with torch.no_grad():
      model.eval()
      output = model(input_tensor)

      if output.item() < 0.5 and target.item() == 0:
        correct += 1
      if output.item() > 0.5 and target.item() == 1:
        correct += 1

      total += 1

  accuracy = (correct / total) * 100
  print('accuracy: {}%'.format(accuracy))

