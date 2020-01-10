
# Pulsar Star Classification with Logistic Regression

Classifies Pulsar Star candidates from the HTRU2 dataset using
Logistic Regression to 97% accuracy

The dataset was split 70% training and 30% for evaluation of the model
over 2 epochs where the model was exposed to the entire training set

## EDIT
After reviewing the data and noticing the huge difference in positive
classifications (actual pulsars), I tested the model with only positive
candidates. The model's accuracy shifted to ~63% with new values

I believe this same reason also leads to the spikes in the loss visualization
in the image `loss.png`. Balancing of the data should fix this issue and lead
to better accuracy but may make the model easier to underfit

## Usage
 - `pip install -r requirements.txt`
 - This project uses `pytorch`, visit their website for installation
   instructions specific for your setup
 - Extract `pulsars.zip` (the dataset) to a `pulsar_stars.csv`
 - `python main.py train` to train the model which saves as
   `model.pth` when finished, or `python main.py test` to test
   the model

## Outputs
 - `loss.png` -- Created when training, visualizes training during
   the iterations
 - `model.pth`  -- saved state dict of model to be used for
   predicting new data

