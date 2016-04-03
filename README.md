# MachineLearningAssignment1
Georgia Tech Machine Learning assignment 1


Code url: https://github.com/dmmiller612/MachineLearningAssignment1

# Instructions for ml assignment 1 AND 3

If wanting to run the server locally, instead of just using the Weka models located in /src/main/resources/models, there are a couple of dependencies needed: Maven and Java.

1. This uses java 1.7, but should work with 1.8 as well. For the JRE, `sudo apt-get install default-jre` . For the jdk, `sudo apt-get install default-jdk`.

2. This uses Maven 3.x . To Install maven 3, use `sudo apt-get install maven`.

3. Go to the root of the assignment code repository and type: `mvn clean package` into the command line. Then type `java -jar target/derek-assignment-1-0.1.0.jar`. Once running the jar, all of optimal models will start to run against the test datasets of both the Car Evaluation and Census dataset. This is here just to make it easier to visualize, so that you do not have to use the rest api. If you want to use the rest api, see documentation below.

4. IF three does not work, it is because the plugin did not properly install. Running this command should do the trick inside of the students-filters-master
"mvn install:install-file -Dfile=filters-0.0.1-SNAPSHOT.jar -DgroupId=filters -DartifactId=filters -Dversion=0.0.1-SNAPSHOT -Dpackaging=jar"


# Navigating the Source Code 

src/main/java/com/derek/ml/controllers => contains the rest endpoints.

src/main/java/com/derek/ml/services => contains all of the logic and configuration of weka models. ClusterService -> k-means and EM, FeatureReductionService -> ICA, PCA, RP, CFS, KNNService -> KNN, NeuralNetworkService -> Neural Network, DecisionTreeService->Decision Trees (boosted an unboosted), SVMService->SVM

src/main/java/com/derek/ml/models => DTO passing layers




# Navigating the Resources 

src/main/resources/csv => Contains all of the initial csv files used (Arffs are only used for the models, however)

src/main/resources/arffs => Contains all of the arffs used. car_train.arff and car_test.arff are the training and testing instances for the car evaluation dataset. census.arff and censusTest.arff are the training and testing instances for the Census dataset. 

src/main/resources/models => Contains several models used for the supervised learning analysis. If you don’t want to run the code locally, you can just use these models against the training and test arffs listed above.





# Using the Rest API (Optional)

I thought I would just add this to show the code I used for experimentation with Weka. I used the api, so that I could do multiple concurrent requests.

Universal Query parameters: fileName : {Car, Census, CarBin, CensusBin}, testType : {CrossValidation, TestData, Train}

__________________
| Cluster        |
-------------------

Endpoints: /kMeans and /em
Query Params => clusters : int, distances : {Euclidean, Manhatten}, iterations: int, featureSelection: {ICA, PCA, RP, CFS};

----------------------
| Feature Reduction |
----------------------

Endpoints: /featureReduction/pca /featureReduction/ica /featureReduction/rp /featureReduction/cfs
_________________
|Decision Trees: |
—————————————————

Endpoint: /decisiontree

Query Params => minNumObj : int, boost : boolean, confidence : String, treeType : {ID3, J48}

Example Requests:
http://localhost:8080/decisiontree?fileName=Car&testType=TestData&minNumObj=2&confidence=.25
http://localhost:8080/decisiontree?fileName=Census&testType=CrossValidation&minNumObj=2&confidence=.25
http://localhost:8080/decisiontree?fileName=Car&testType=CrossValidation&minNumObj=2&confidence=.25&boost=true //with boosting

Using incremental testing example:
http://localhost:8080/decisiontree/test?fileName=Car&testType=TestData&minNumObj=2&confidence=.25&boost=true

________________
| KNN           |
————————————————

Endpoint: /knn

Query Params => k : int, boost : boolean, treeTypes {BallTree, CoverTree, Linear}, useFeatureSelection : boolean (applies only to Census file)

Examples : 
http://localhost:8080/knn?fileName=Car&testType=TestData&k=3
http://localhost:8080/knn?fileName=Census&testType=TestData&k=5&featureSelection=true
http://localhost:8080/knn?fileName=Census&testType=TestData&k=5&boost=true

Using incremental testing example:
http://localhost:8080/knn/test?fileName=Census&testType=TestData&k=5

__________________
| ANN             |
——————————————————

Endpoint: /neuralnetwork

Query Params => hiddenLayers : int, epochRate : int, featureSelection : boolean (applies only to Census file)

Examples:
http://localhost:8080/neuralnetwork?fileName=Car&testType=TestData&hiddenLayers=10&epochRate=500
http://localhost:8080/neuralnetwork?fileName=Census&testType=TestData&hiddenLayers=5&epochRate=500&featureSelection=true

Using incremental testing example:
http://localhost:8080/neuralnetwork/test?fileName=Car&testType=TestData&hiddenLayers=10&epochRate=500


______________
|SVM          |
——————————————

Endpoint: /svm

Query Params => kernelType : {Polynomial, RBF, Sigmoid, Linear}


Examples:
http://localhost:8080/svm?fileName=Car&testType=TestData&kernelType=Polynomial
http://localhost:8080/svm?fileName=Census&testType=TestData&kernelType=RBF
http://localhost:8080/svm?fileName=Census&testType=TestData&kernelType=Sigmoid
http://localhost:8080/svm?fileName=Census&testType=TestData&kernelType=Linear


# MODELS

The model names contain the parameters that were used, fileName, and algorithm name.

Decision Tree Naming Convention: decisionTree + minNumObj + Boosted + confidence + fileName + .model
Example: decisionTree-minNumObj=100-Boosted=false-C=0.25-file=Census.model

KNN Naming Convention: KNearestNeighbor + k + fileName + .model
Example : KNearestNeighbor-k=20-fileName=Car.model

ANN Naming Convention: ANN + hiddenLayers + epochRate + FileName + (Optional) featureSelection + .model
Example : ANN-hiddenLayers=10-epochRate=250-FileName=Census.model

SVM Naming Convention: SVM + kernelType + FileName + .model
Example : SVM-KernelType=Linear-FileName=Car.model
