

# Classifying an imbalanced dataset 95% vs 5% class ratio.
For specific dataset Sourced deals vs industry deals. Total 67K deals out of which 3K are sourced deals and 64K are industry deals using large number of features.

Performance metric: Recall for top 20% score i.e. accuracy/recall at k
1. Random Forest Classifier with class weights
    - class weights only improves perforamnce a little.
    - Test set performance keeps improving (a very little though) even when we overfit the model. i.e. large gap between train and test accuracy when we overfit.

2. Oversample the minority class using SMOTE
    - Oversampling the minority class doesnot improve the results.

3. Using BalancedRandomForestClassifier
    - It creates each tree on bootstrapped sample of minority class and equal size sample of majority class.
    - So each tree is trained on balanced sample of the dataset.
    - Though test accuracy we get is similar to Random Forest Classifier. But the model is more robust and less overfitting.


![hp_tune_comparisionx.png](../output/run3/hp_tune_comparisionx.png)
