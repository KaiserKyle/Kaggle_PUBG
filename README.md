# Kaggle_PUBG

R scripts for PUBG Finish Placement Prediction competition on Kaggle

These scripts were used in the aforementioned Kaggle competition, which can be found here:
https://www.kaggle.com/c/pubg-finish-placement-prediction/
This was also used as a final project for one of my graduate classes, PUBG Final Report.docx is a copy of the paper I turned in which has all the details of the project.

randomforest.r - Contains a Random Forest implementation of the solution. There are three main components to the pre-processing done:
  1) Partition data into 'singles', 'doubles', and 'team' matches. Each match type plays differently, so we create a seperate model for each
  2) Scale data by team ('group' in the script) in the 'doubles' and 'team' matches. This allows us to know who the top performers are on a team and how big the skill gap is from top player to lowest player on a given team
  3) Scale data by match. Each match is different, so the raw stats aren't as helpful. Scaling by match normalizes all match and makes them more comparable.
  
xgboost.r - Contains the same basic logic for preprocessing, but applies an XGBoost model instead for slightly better accuracy. 
