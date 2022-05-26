# CNVABNN
A method for CNV detection using neural networks and the AdaBoost framework

# Table of Contents
1. Introduction
2. Requirement/environment
3. Usage of CNVABNN
4. Detection performance of CNVABNN

## introduction
CNVABNN is a cnv detection method based on RD strategy for single sample. The project provides the complete implementation code and a partial dataset. In the test dataset, there are two parts of samples, one part is simulated sample and the other part is real sample. Specific details of the usage can be found in the usage section of CNVABNN.

## Requirement/environment
This code was tested with Matlab R2020b and the machine configuration used for testing was as follows：
system: windows
CPU：Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz 
Before running CNVABNN, BAM file needs to be preprocessed, and the corresponding code can be found in ‘code/main_simulation1.py’
## Usage of CNVABNN
### step1: Modify configuration
Before training the model, the path of the data needs to be modified. The path of data1-data6 used in train.m needs to be modified. The imported data are from the partial simulated dataset, which can be found in the train folder.

### step2: Train
Run train.m to train the neural networks based on partial simulation data. Moreover, the training model needs to be saved for testing purposes, so the saving path of the corresponding parameters needs to be modified, and the parameters involved include all weak classifiers(BP_Ada_1~BP_Ada_3), combination weights(ak), and the number of weak classifiers(k).

### step3: Detection of CNV in the simulation datasets
Detection of CNVs in the simulated sample requires importing the training model in Ada_test_sim.m, all weak classifiers(BP_Ada_1~BP_Ada_3), combination weights(ak), and the number of weak classifiers(k).The results obtained using the trained neural network are the probabilities of the four copy number states. By comparing the magnitude of the probabilities, \The structure of the neural network in this project is shown in the following figure:
![](img/network.jpg)
In addition, this project incorporates the Adaboost algorithm to improve the detection performance of the neural network. In the test of simulated samples, we used a total of 300 samples, some of which can be found in SimulationData folder. The output of Ada_test_sim.m is the precision and sensitivity of the CNV prediction. If you want to extract the location and type of variation, you need to modify line 166 to output numbers of variant bin to a file.

### step4: Detection of CNV in the real datasets
The detection of the real datasets of CNVs using CNVABNN similarly to step 3. Running Ada_test_real simply requires importing the training model and modifying the path. Among them, we give the three real datasets used for testing, NA19238,NA19239,NA19240, which can be found in the RealData folder.

## Detection performance of CNVABNN
CNVABNN achieves good performance on low coverage datasets and is also adaptable at higher coverage. The performance comparison with peer methods is as follows:
![](img/performance.jpg)
