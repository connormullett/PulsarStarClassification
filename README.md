
# Pulsar Star Classification with Logistic Regression

Classifies Pulsar Star candidates from the HTRU2 dataset using 
Logistic Regression to 97% accuracy

The dataset was split 70% training and 30% for evaluation of the model
over 2 epochs where the model was exposed to the entire training set

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

