# Time_Series_Classification_Deep_Learning_Improvements
Trying to reproduce and improve the original UCR experiments in the following link:
https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline

## Used datasets
- Adiac: https://www.timeseriesclassification.com/description.php?Dataset=Adiac
- GunPoint: https://www.timeseriesclassification.com/description.php?Dataset=gunpoint
- InlineSkate: https://www.timeseriesclassification.com/description.php?Dataset=InlineSkate
- Coffee: https://www.timeseriesclassification.com/description.php?Dataset=Coffee
- MedicalImages: https://www.timeseriesclassification.com/description.php?Dataset=MedicalImages

## Reproducing the results
The notebooks ending with '_Reproduced.ipynb' contain the code for reproducing the paper's results for each model. Ensure that the dataset files are in the same directory as the notebook and then simply run the code. No changes were made to the original code except for the necessary minor debugging required to ensure the code runs on modern TensorFlow / Keras. 

## Improving the results
The notebooks ending with '_Improved.ipynb' contain the code with various improvements for each model, along with the corresponding descriptions, results, and visualizations.
