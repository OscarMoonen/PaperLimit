# PaperLimit

#RUNNING THE MODEL
#1.
#.IPYNB model can be openend in Jupyter notebooks
#.PY model can be opened with Python
#2.
Install the packages that are annotated in the code
3.
Set Parameters under "Parameters Model"
4.
Data will be stored as .npz files


#READING THE DATA
1.
Open Python or Jupyter Notebooks, with NumPy installed and imported
2.
Put the .npz in the working directory
3.
use the following command to open the .npz files
--> data = np.load('fileName.npz')
and the following command to retrieve the tracked  statistics per generation
--> statistic = data["statistic"]
every statistic retrieved has the shape of a matrix with the size of [repeats x generations]

The following statistics can be retrieved from the file:
#For each generation:
combinedPO = average payoff 
combinedQW = average questions worked
combinedSS = average sample size
combinedAP = average accept probability
combinedRWP = average random walk probability
combinedAA = average authors per paper
combinedDRAW = average drawer size
combinedPUB = average publications
TP = True positives
TPP = True positives published
FP = False positives
FPP = False positives published
TN = True negatives
TNP = True negatives published
FN = False negatives
FNP = False negatives published
#For the last generation:
finalAP = accept probabilities of the last generation
finalRWP = random walk probabilities of the last generation
finalSS = sample sizes of the last generation
finalPN = confusion matrix of the last generation
