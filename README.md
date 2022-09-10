# SFD_Problem (Seattle Fire Department Problem)
<br>
<img src="https://github.com/neuhart/SFD_Problem/blob/main/Hourly%20Call%20Volume%201st%20week%20March'19.png" height="350" width="550" align="center"><br>


## Project Summary
The Seatte Fire Department Dataset contains 171M 911 dispatches (emergency calls) with several features such as location, datetime or address.
The goal was to predict the number of hourly emergency calls for the year of 2019. 

## Data
The main code is provided in the Jupyter Notebook <a href="https://github.com/neuhart/SFD_Problem/blob/main/main.ipynb">main.ipynb text</a>. <br>
The code is tested and runs both offline and in google collabs. Since requested, each step of the project is save in an individual file:<br>
- <a href="https://github.com/neuhart/SFD_Problem/blob/main/Preprocessing.py">Preprocessing</a> <br>
- <a href="https://github.com/neuhart/SFD_Problem/blob/main/feat_eng.py">Feature Engineering text</a><br>
- <a href="https://github.com/neuhart/SFD_Problem/blob/main/training.py">Training</a><br>
- <a href="https://github.com/neuhart/SFD_Problem/blob/main/testing.py">Testing</a><br>

The tuned Deceision Tree model is saved in <a href="https://github.com/neuhart/SFD_Problem/blob/main/dec_tree_pipe.joblib">Model</a>.

## Design and Assumptions
- preprocessed the data by getting rid of null values, redundant columns
- split up the Datetime column to isolate each time unit (years, months, days, etc.)
- created several additional features (e.g.: season feature).

(As hinted) I used the Boosted Decision Tree regression model with a maximal depth of 5 and a maximum number of estimators of 300. I've already worked with Decision Trees before which made me go for a Decision Tree instead of Gradient Boosting. As suggested I restricted the training set to the years 2014-2018. I used only random subset of 100000 instances due to computational limitations. <br>
I used the sklearn pipeline framework to create a pipeline for additional preprocessing like standardization or one-hot-encoding of categorical variables.

## Results

As evaluation metric, I used both the R2-score and the MSE. <br>
Evaluated on the training data (random subset of 100k instances), I obtained a MSE of 15.87 and a R2 score of 0.387. <br>
Evaluated on the test data (2019), I obtained a MSE of 17.54 and a R2 score of 0.325.

- Training MSE Loss: 15.872172597897624 
- Test R2 score 0.3874855409474047
- Test MSE Loss: 17.53968216023331 
- Test R2 score 0.3248216803738089
