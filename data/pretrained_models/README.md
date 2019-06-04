# dpUGC: Learn Differentially Private Representation for User Generated Contents

## Introduction to the model

## Use dpUGC to train dp embedding models:
Note that: because of the myPersonality data is a register data, we can only provide the pre-trained embedding model. 
Thus, if user want to train the model again, they have to train on their own data or on Text8 data as we provided.
*  How to run:
> 01.run_train_dp_embedding.sh
* Expected output:
> A pre-trained embedding model at <OUTPUT>.

## Reproduce Experiments:
### 1. See semantic spaces change when train the embedding with/without privacy guarantee.

*  How to run:

> 02.run_changes_in_semantic_space.sh

* Expected output:

### 2. Evaluate the benefit of dpUGC models on down-stream tasks: linear regression
This experiment run the linear regression task on personality regression by using public embedding and private-embedding
, which was trained from the register myPersonality data.

*  How to run:
> 03.run_evaluation_regression_task.sh

* Expected output:

