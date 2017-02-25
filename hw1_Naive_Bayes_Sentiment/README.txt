usage: python NB_sentiment.py [-h] [-path PATH] [-train TRAIN TRAIN] [-test TEST]
                       [-tmp_path TMP]

optional arguments:
  -h, --help          show this help message and exit
  -path PATH          [optinal arg] file path for token directory, if no path argument program will use default path under "/user/cse842/SentimentData/tokens/"
  -train TRAIN TRAIN  select two fold for training, input 1,2,3, example -train 1 2  will use 1st fold and 2nd fold to train model
  -test TEST          select fold for test, input 1,2,3
  -tmp_path TMP       [optional arg] file path for temporary saved files, if no argument then
                      program will not save the trainin result
					  
Result	(also example)				  
<32 arctic:~ >python "/user/dunan/CSE842/NB_sentiment.py" -train 1 2 -test 3
The accuracy is 0.784946236559
<33 arctic:~ >python "/user/dunan/CSE842/NB_sentiment.py" -train 2 3 -test 1
The accuracy is 0.771241830065
<34 arctic:~ >python "/user/dunan/CSE842/NB_sentiment.py" -train 1 3 -test 2
The accuracy is 0.757575757576

So average accuracy is 0.771255

Save File Format:
if you input the tmp_path argument
The saved file will save in pkl format
The line start with s is the token, and the line above token is the probability.
In the file program saved 4 parameters, positive prior, positive_token_probability, negative prior, and negtive_token_probability, since it is organized by python module pickle, to figure out some specific parameter may need extra effort

Save File Name
So the name of model saved file is determined by the -tmp_path argument. 