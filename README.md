# nlp_legislation_prediction
Class project to predict party affiliation of of legistators from legislation text.

This is part of a larger idea, to be able to simulate the effects of legislation. But the first step is learning how to process legislative text, which is highly formalized. Then run some sort of categorization/predicition on it.

The goal for this project is to take a set of legislative text. Extract the changes or new legislative text. Using the party affliation of the author(s) as the label. 

Train by extending using some base NLP model.

Also within scope may be to predict 
a) did the legislation make it out of committee
b) did the legislation pass
c) did the legislation get vetoed


Data and model checkpoints are saved using git lfs. To clone this 
repo, you will need to have git lfs installed on your system.

[https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage](Git LFS install)


T-SNE
Clone : https://github.com/mxl1990/tsne-pytorch 

Run the t-sne-exporter notebook to export the text files this tool wants

# Project layout
Each section describes on the main subdirectories

## data_preparation 

### acquisition
* code for data acquisition should be run in this order
  * download_raw.py
  * extract_text.py
  * extract_bill_data.py

download_raw.py assumes you have an api key for legiscan.org. This api key is presumed to be in a directory name 'secrets'
off of the project root directory and in a file named legal_scan_key.txt. The file just contains your api key.

The extracted data contains the following fields
* summary_bill_(session_id)_(bill_id).json
  * text -- the title of the bill
  * status -- a numeric enumeration of the bill's status (taken from the legiscan api)
  * party -- 0-3 inclusive. 0 = all democratic party sponsorship, 1 = all republican sponsorship, 3 = mixed, 2 = independent (not present in the data)
* summary_bill_text_(session_id)_(bill_id).json
  * text -- the full text of the bill (with some clean up)
  * status
  * party

### tokenization
After the data has been downloaded and processed, then we need to produce the tokenized version of the text fields
* the notebooks for tokenization
  * convert_to_bert_token.ipynb -- use BERT to tokenize the summary_bill_(session_id)_(bill_id).json data
  * convert_to_bert_token_full_text.ipynb -- use BERT to tokenize the summary_bill_text_(session_id)_(bill_id).json data



### exploration
High level summaries of the text, tokenized text, status and party data is presented in the data_exploration notebooks
* data_exploration.ipynb -- data exploration of the summary_bill_(session_id)_(bill_id).json data
* data_exploration_full_text.ipynb --  data exploration of the summary_bill_(session_id)_(bill_id).json data
## data

### data/raw
Raw data is written to the data/raw directory. This is where the download_raw.py and extract_text.py write their data.

The tar.gz file here can be extracted for the previously downloaded data.

### data/extracted
Extracted (or summary data) is written to the data/extracted directory. The files here contain only the fields from the raw data that 
are expected to be used in training.

The tar.gz file here can be extracted for the previously summarized data.
### data/tokenized
The output from the convert_to_bert*.ipynb notebooks is written here.

The tar.gz file here can be extracted for the previously tokenized data (in pickle format with gzip compression)


## training
Source code for the pytorch training lives here.
* train.ipynb -- the first attempt at training on the summary_bill_(session_id)_(bill_id).json
* project_function.py -- has function and models used for later model trainings where I decided to extract common code
* 

