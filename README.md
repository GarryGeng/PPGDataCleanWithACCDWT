# PPG Project

This project contains methods for processing PPG (Photoplethysmography) data.

## Methods

There are two main methods for processing PPG data:

1.  **TinyPPG:**  Detect and delete unusable data through TinyPPG Model
2.  **ACC Data:** Detect and delete unusable data through ACC data

## Current Working on

- **/ACC_DWT/EDA_ACC_Hole.py** :  Currently being used for EDA (Exploratory Data Analysis) with ACC data, focusing on handling holes in the data.
- **/ACC_DWT/EDA_ACC_Sliced.py** :  Currently being used for EDA with ACC data, focusing on sliced data.


## Legacy

- **/TinyPPG/EDA_DwtTny.py:** Analyzes PPG signals using DWT and the TinyPPG model to detect invalid data segments and visualize signal quality.
- **/TinyPPG/EDA_DwtTny_OLD.py:** Legacy version for performing EDA on PPG data from a text file with DWT and the TinyPPG model.
- **/TinyPPG/EDA_ACC_SlicedTny.py:** Processes sliced PPG and ACC data by cleaning problematic segments with change detection and applying DWT for analysis.
- **/ACC_DWT/EDA_ACC_Sliced_Old.py:** Legacy script that applies change detection and DWT on ACC and PPG data, visualizing signal quality.
- **/ACC_DWT/EDA_ACC_Hole_Old.py:** Legacy script for detecting holes in ACC data and processing PPG signals via DWT, including SQI computation.
