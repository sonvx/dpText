# dpUGC: Learn Differentially Private Representationfor User Generated Contents 
* Paper: https://arxiv.org/abs/1904.10454
## How to cite:
```
@InProceedings{Vu:CiCLing:2019b,
	author    = {Xuan-Son Vu, Son N. Tran, Lili Jiang},
	title     = {dpUGC: Learn Differentially Private Representationfor User Generated Contents},
	booktitle   = {Proceedings of the 20th International Conference on Computational Linguistics and Intelligent Text Processing, April, 2019},
	year      = {2019},
	location = 	{La Rochelle, France}
}
```

## How to train privacy-guarantee embedding models on new dataset.
* This is for training private embedding on new data:
> cd codes/ <br>
> ./10.run_train_dp_embedding.sh

## How to prepare the environment to run the code:
Create a python environment using [virtualenv](https://docs.python.org/3/library/venv.html) 
or [anaconda](https://www.anaconda.com/distribution/), 
then run this command in that environment:
> pip install -r requirements.txt

## How to reproduce experiments in the paper:

### Experiment 1: Semantic Changes
* How to run:<br>
> cd codes/ <br>
>
> ./01.run_changes_in_semantic_spaces.sh
* Expected outputs: see the evaluation results from the console (see codes/images/results_fig2.png). 
The results should be similar to **Figure 2** in the paper.

* Note:
It takes time to train the embedding model so we already extracted the top similarity 
words to run the evaluation. If it's needed, users are able to train again from the text8 corpus
and select the top similar words and run this evaluation again.

### Experiment 2: Linear Regression Task.
* How to run:
> cd codes/ <br>
>
> ./02.run_evaluation_regression_task.sh

* Expected outputs: printed in the console (see codes/images/results_table3.png). Evaluation results similar to **Table 3** in the paper. 
Note that the privacy-budget column was stated after the training embedding with differential privacy, 
i.e., one according privacy-budget (\delta, \epsilon) is calculated at each checking point.
