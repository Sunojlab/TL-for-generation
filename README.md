The model is implemented using PyTorch deep learning framework and fast.ai library. All the calculations are run using the Google Colab Pro. The fast.ai library uses pre-trained language models, performs fine-tuning, in the following three steps,
(1) Data pre-processing
(2) Use the pre-trained weights to create a language model (LM) that can be used for fine-tuning on the given dataset
(3) Create models such as classifiers and/or regressors using the pre-trained or fine-tuned LM.

Different steps involved in using Fastai’s ULMFiT 

Step 1: Data Pre-processing

First, a “databunch” is created for the given task that can be for a language modeling or a classification/regression problem. The fast.ai provides simple functions to create the respective databunch and automatically performs the pre-processing of text. 
 
Following two types of databunch can be created,
(1) TextLMDataBunch: it creates a databunch for the task of language modeling, and doesn’t require separate data labels. The data is processed in a manner such that the model learns to predict the next word given the previous words in a sentence.
(2) TextClasDataBunch: it creates a databunch for the task of classification/regression, and requires labeled data. 

The fast.ai databunch can take the input in the form of dataframes. The train and validation (or test) dataframes has to be provided to build the databunch. The columns containing text and labels have to be specified. For building a databunch for language model, the label column can be ignored. The two main tasks when preparing the text data for modeling are: (a) tokenization, which splits the text into tokens, and (b) numericalization, that assigns a unique integer to each token. Either the fast.ai’s default tokenizer and numericalizer can be used or a custom tokenizer with specific vocabulary for numericalization can be passed. Once the pre-processing is done, the databunch can be saved. It can then be loaded as and when required.
 
Step 2: Creating Language Model

Language model fine-tuning:
This is the first step in training, where the pre-trained LM weights are used for fine-tuning on target data. Fast.ai provides an easy way of creating a “learner” (language_model_learner) for language model training. It requires LM databunch and a pre-trained model as input. AWD-LSTM is the model architecture used to train LM. There are other architectures available as well (e.g. transformer decoder, transformerXL). The ‘config’ argument can be used to customize the architecture. The ‘drop_mult’ hyperparameter, can be tuned to set the amount of dropout, a technique used for regularization. The pre-trained weights and the corresponding vocabulary can be passed to ‘pretrained_fnames’ argument.

Training the model:

Learning rate (lr) is one of the most important hypeparameters in the training of a model. The fast.ai’s utility ‘lr_find’ can be used to search a range of lr and the plot of lr versus loss is used to identify the lowest loss and choose the lr one magnitude higher than that corresponds to lowest point. The LM model then can be trained with this lr using ‘fit_one_cycle’. The fit_one_cycle takes the lr and number of epochs as arguments.
The encoder of the trained model that has learned the specifics of the language can be saved and later used for other downstream tasks (classification/regression). 
 
Step 3: Creating the Regressor/Classifier

In the first step, a TextClasDataBunch is created using the vocabulary of the pre-trained or fine-tuned LM. This is done to make sure that the LM and regressor have the same vocabulary. The batch size ‘bs’ argument can be set (32, 64, 128, 256, etc.,) according to the system memory available.
 
In the second step, a learner “text_classifier_learner” is created for the regression task. It takes the databunch created in the first step as input. The encoder of the pre-trained/fine-tuned model saved in step 2 can be loaded. Then the same procedure can be followed for finding the lr and training the model. 

Pre-trained weights

The data, code and pre-trained weights for training the LM on 1 million SMILES strings is provided at https://github.com/Sunojlab/Transfer_Learning_in_Catalysis

Datasets

The dataset used for generation and regression is provided in the ‘Data’ folder. 
1.	‘alcohol-smiles.csv’ contains the SMILES of 37 training set alcohols used for LM fine-tuning and generation
2.	‘740-reactions-yield.csv’ contains the SMILES of previously reported 740 deoxyfluorination reactions used for regressor fine-tuning
3.	‘generated-alcohols.csv’ contains the SMILES of 75 newly generated alcohols
4.	‘new-reactions.csv’ contains the SMILES of new set of 1500 reactions.

Code

All the codes are provided as notebooks that can be directly run on Google Colab 
1.	All the notebooks for LM and regressor fine-tuning are present in the ‘Notebooks’ folder
2.	The notebook for the fine-tuning the LM followed by generation is ‘Fine-tuning of LM for generation’
3.	The notebook for fine-tuning the regressor for yield prediction is ‘Fine-tuning of regressor’

References

•	https://doi.org/10.3390/info11020108
•	https://github.com/XinhaoLi74/MolPMoFiT
•	https://github.com/marcossantanaioc/De_novo_design_SARSCOV2
