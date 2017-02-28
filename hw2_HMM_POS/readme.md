# How to run this code

usage: 

> python HMM_POS.py [-h][-lambda SMOOTH] [-k K] training_path test_path test_truth_path


positional arguments:
​		training_path    the training file path for HMM POS
​		test_path        the test file path for HMM POS, no tags
​		test_truth_path  the test thruth file path for HMM POS

optional arguments:

​		-h, --help       show this help message and exit
​		-lambda SMOOTH   smooth parameter
​		-k K             hwo many result want to output, other wise obly report accuracy

Example:
>python "/user/dunan/CSE842/HMM_POS.py" "/user/cse842/POSData/wsj1-18.training" "/user/cse842/POSData/wsj19-21.testing" "/user/cse842/POSData/wsj19-21.truth"

This will just using 1 as smooth parameter and do not output any tags of sentences

> python "/user/dunan/CSE842/HMM_POS.py" "/user/cse842/POSData/wsj1-18.training" "/user/cse842/POSData/wsj19-21.testing" "/user/cse842/POSData/wsj19-21.truth" -lambda 0.5 -k 20

This will use 0.5 as smooth parameters and will print first 20 sentences results