# Neuroimage-benchmark

This package serves as a framework to use the ADNI and UKBB data sets for cognitive assessment regression and sex classification.
However, this system can be easily configured to predict other target labels.Due to the size of the MRI and semi-private nature
of the datasets. The data is not available for distribution and inclusion in this package. However, for grading and verification
purposes we will provide jupyter notebooks and commands that were used to yield our results

## Most importantly for Metric Verificiation
We provide the jupyter notebook **./notebook/model_metrics_verification.ipynb** to verify the results stated in our report.
This is done by loading all the report featured models, saved on the server, organize the data for prediction. Then
predictions are recorded and metrics calculated and reported at the end. **All metrics featured in our report can be verified
in this notebook.**

## Prediction Figures
We provide the jupyter notebook **./notebook/model_output_analyzer.ipynb** to further verify the predictions stated in our report.
In this notebook our system saves the predictions, true labels, and loss metrics in csvs that are paired with the saved model.
In this notebook we utilize these saved csv predictions for plotting and further metric calculation
**All figures regarding our predictions featured in our report can be verified in this notebook.**

## Data Exploration
We provide the jupyter notebook **./notebook/adni_data_exploration.ipynb** to verify the data stated in our report.
**All figures regarding our data featured in our report can be verified in this notebook.**

## Next are the commands that we used to carry out the experiments. 
These consisted of parameter configurations loaded into the python main.py file. The below commands could be replaced to 
use TCNN, Densenet, or Tiny VGG if -ma is substituted by tcnn, densenet, and vgg respectively.

### Example for baseline experiments for Resnet
- python main.py -g 1 -bs 4 -tc MMSE -ma resnet -ilr 0.005 -wd 0.0001 -en resnet_baseline -l mse -es 15 -do 0.4
This command if ran on our server would execute the baseline experiment for resnet
### Example for pre-training for Resnet
- python main.py -g 1 -bs 4 -tc abel -trp /lfs1/pjtoral/imbalanced-regression/data/classification/sex/df_ukbb_train.csv 
-tep /lfs1/pjtoral/imbalanced-regression/data/classification/sex/df_ukbb_test.csv -ma resnet -ilr 0.01 -wd 0.0001 
-en resnet_transfer_learning -l bce -es 15 -do 0.4
This command if ran on our server would execute the pre-training experiments for resnet
### Example for fine-tuning for Resnet
- python main.py -g 1 -bs 4 -tc MMSE -ma resnet -ilr 0.005 -wd 0.0001 -en resnet_fine_tuning -smp ./resnet_tl__2023-11-29_11:48:32/save/
-l mse -es 15 -do 0.4
This command if ran on our server would execute the fine-tuning experiments for resnet