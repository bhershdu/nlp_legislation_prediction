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
