package com.derek.ml.services;

import com.derek.ml.models.ML;
import com.derek.ml.models.NeuralNetworkModel;
import com.derek.ml.models.Options;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

@Service
public class NeuralNetworkService {

    @Autowired
    private FileFactory fileFactory;

    @Autowired
    private EvaluationService evaluationService;

    @Autowired
    private FeatureReductionService featureReductionService;

    @Autowired
    LoadData loadData;

    public String handleNeuralNetwork(NeuralNetworkModel neuralNetworkModel) throws Exception{
        FileFactory.TrainTest data = fileFactory.getInstancesFromFile(neuralNetworkModel.getFileName(), new Options(neuralNetworkModel.isFeatureSelection()));
        MultilayerPerceptron multilayerPerceptron = handleClassification(data.train, neuralNetworkModel);
        return handleEvaluation(multilayerPerceptron, neuralNetworkModel, data);
    }

    public MultilayerPerceptron handleClassification(Instances data, NeuralNetworkModel neuralNetworkModel) throws Exception{
        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();
        multilayerPerceptron.setTrainingTime(neuralNetworkModel.getEpochRate());
        multilayerPerceptron.setHiddenLayers(Integer.toString(neuralNetworkModel.getHiddenLayers()));
        multilayerPerceptron.buildClassifier(data);
        return multilayerPerceptron;
    }

    public String handleEvaluation(MultilayerPerceptron multilayerPerceptron, NeuralNetworkModel neuralNetworkModel, FileFactory.TrainTest data) throws Exception{
        if (neuralNetworkModel.getTestType() == ML.TestType.TestData){
            return evaluationService.evaluateData(data.train, multilayerPerceptron, data.test) + "\n \n" + multilayerPerceptron.toString();
        } else if (neuralNetworkModel.getTestType() == ML.TestType.Train){
            return evaluationService.evaluateData(data.train, multilayerPerceptron, data.train) + "\n \n" + multilayerPerceptron.toString();
        }
        else {
            return evaluationService.evaluateData(data.train, multilayerPerceptron, 6) + " \n \n" +  multilayerPerceptron.toString();
        }
    }

    public String handleSplitData(NeuralNetworkModel neuralNetworkModel, int num, String retString) throws Exception{
        if (num <= 100){
            retString += "Amount " + Integer.toString(num) + "\n";
            FileFactory.TrainTest data;
            if (neuralNetworkModel.getFileName() == ML.Files.Census){
                data = fileFactory.handlePublicCensus(num, new Options(neuralNetworkModel.isFeatureSelection()));
            } else {
                data = fileFactory.handlePublicCar(num);
            }
            MultilayerPerceptron multilayerPerceptron = handleClassification(data.train, neuralNetworkModel);
            Instances d;
            if (neuralNetworkModel.getTestType() == ML.TestType.Train){
                if (neuralNetworkModel.getFileName() == ML.Files.Car){
                    d = fileFactory.handlePublicCar(0).train;
                } else {
                    d = fileFactory.handlePublicCensus(0, new Options(neuralNetworkModel.isFeatureSelection())).train;
                }
            } else {
                d = data.test;
            }
            return handleSplitData(neuralNetworkModel, num==1 ? num+9 : num+10, retString + "\n \n" + evaluationService.evaluateData(data.train, multilayerPerceptron, d));
        }
        return retString;
    }

    public void createModel(NeuralNetworkModel nn) throws Exception{
        FileFactory.TrainTest data = fileFactory.getInstancesFromFile(nn.getFileName(), new Options(nn.isFeatureSelection()));
        Classifier cls = handleClassification(data.train, nn);
        loadData.saveModel(cls, getString(nn));
    }

    public String getModel(NeuralNetworkModel nn) throws Exception {
        FileFactory.TrainTest data = fileFactory.getInstancesFromFile(nn.getFileName(), new Options(nn.isFeatureSelection()));
        Classifier cls = loadData.getModel(getString(nn));
        nn.setTestType(ML.TestType.TestData);
        return handleEvaluation((MultilayerPerceptron) cls, nn, data);
    }

    public String neuralNetworkWithReduction() throws Exception{
        FileFactory.TrainTest censusTrainTest = fileFactory.getInstancesFromFile(ML.Files.CensusBin, new Options());
        Instances pcaCensus = featureReductionService.applyPCAFilter(censusTrainTest.test, 30);
        Instances icaCensus = featureReductionService.applyICA(censusTrainTest.test, 30);
        Instances rpCensus = featureReductionService.applyRP(censusTrainTest.test, 30);

        NeuralNetworkModel neuralNetworkModel = new NeuralNetworkModel();
        neuralNetworkModel.setEpochRate(500);
        neuralNetworkModel.setHiddenLayers(4);

        censusTrainTest.train = pcaCensus;
        String one = handleEvaluation(handleClassification(pcaCensus, neuralNetworkModel), neuralNetworkModel, censusTrainTest);
        censusTrainTest.train = icaCensus;
        String two = handleEvaluation(handleClassification(icaCensus, neuralNetworkModel), neuralNetworkModel, censusTrainTest);
        censusTrainTest.train = rpCensus;
        String three = handleEvaluation(handleClassification(rpCensus, neuralNetworkModel), neuralNetworkModel, censusTrainTest);

        return "PCA \n \n " + one + " \n \n \n ICA \n \n" + two + " \n \n \n RP \n \n" + three;
    }

    private String getString(NeuralNetworkModel nn){
        String s = "ANN-hiddenLayers=" + nn.getHiddenLayers() +
                "-epochRate=" + nn.getEpochRate() +
                "-FileName=" + nn.getFileName();
        if (nn.isFeatureSelection()){
            s += "-feature=true";
        }
        return s + ".model";
    }
}
