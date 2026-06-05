
# coding: utf-8

# In[1]:


# This script is responsible for converting the metadata.json files provided in the 
# Deepfake Detection Challenge (DFDC) dataset into a flat CSV format for easier 
# processing during model training. It processes multiple data parts in a loop.

import pandas as pd # Data manipulation library to handle CSV and JSON

# Loop through the available training dataset parts (e.g., parts 0 to 7)
for i in range(8): # Iterate through each part
    # Construct paths for the input JSON and the output CSV for the current part.
    # Note: These paths assume you have placed your Kaggle dataset parts in the labels folder or similar.
    path_json = "/home/albaloshi/Desktop/Deepfake_detection_using_deep_learning-master/Dataset/dfdc_train_part_"+str(i)+"/metadata.json" # Input JSON path
    path_csv = "/home/albaloshi/Desktop/Deepfake_detection_using_deep_learning-master/Dataset/dfdc_train_part_"+str(i)+"/metadata.csv" # Output CSV path
    
    print(path_csv) # Log the CSV path being processed
    print(path_json) # Log the JSON path being processed
    
    # Read the JSON file. It contains video filenames as keys and metadata objects as values.
    read_json = pd.read_json(path_json) # Load JSON into Pandas
    
    # Convert the raw JSON data into a DataFrame and transpose it so that 
    # each video filename becomes a row index and metadata attributes become columns.
    df = pd.DataFrame(read_json) # Create initial DataFrame
    df_2 = pd.DataFrame(df.transpose()) # Transpose: Keys (filenames) become rows
    
    # Save the intermediate transposed data to CSV
    df_2.to_csv(path_csv) # Initial save to file
    
    # Re-read the CSV to clean up column headers and structure
    read_csv = pd.read_csv(path_csv) # Reload for cleanup
    
    # Define descriptive column names for the metadata
    # URI: Filename, label: FAKE/REAL, original: source video if fake, split: train/val
    read_csv.columns = ["URI","label","original","split"] # Assign headers
    
    # Save the finalized CSV without the row index to keep the file clean
    read_csv.to_csv(path_csv,index=False) # Final save without indices
    
    # Display the first 5 rows to verify the conversion was successful
    print(read_csv.head(5)) # Output preview

