package com.derek.ml.services;


import com.derek.ml.models.ML;
import com.derek.ml.models.Options;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.attributeSelection.*;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.IndependentComponents;
import weka.filters.unsupervised.attribute.RandomProjection;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileWriter;

@Service
public class FeatureReductionService {

    @Autowired
    private FileFactory fileFactory;

    public String handlePCAFeatures() throws Exception{
        FileFactory.TrainTest carTrainTest = fileFactory.getInstancesFromFile(ML.Files.Car, new Options());
        FileFactory.TrainTest censusTrainTest = fileFactory.getInstancesFromFile(ML.Files.Census, new Options());
        return ApplyPCA("CAR", carTrainTest.train) + "\n \n \n \n \n" + ApplyPCA("CENSUS", censusTrainTest.train);
    }

    public String handleRandomizedProjectionFeatures() throws Exception{
        FileFactory.TrainTest carTrainTest = fileFactory.getInstancesFromFile(ML.Files.Car, new Options());
        FileFactory.TrainTest censusTrainTest = fileFactory.getInstancesFromFile(ML.Files.Census, new Options());
        return applyRP(carTrainTest.train, 4).toString() + "\n \n \n \n \n" + applyRP(censusTrainTest.train, 4).toString();
    }

    public String handleICAFeatures() throws Exception {
        FileFactory.TrainTest carTrainTest = fileFactory.getInstancesFromFile(ML.Files.CarBin, new Options());
        FileFactory.TrainTest censusTrainTest = fileFactory.getInstancesFromFile(ML.Files.CensusBin, new Options());
        return applyICA(dropClass(carTrainTest.test), 4).toString() + "\n \n \n \n \n" +  applyICA(dropClass(censusTrainTest.test), 4).toString();
    }

    public String handleCFSSubsetEval() throws Exception {
        FileFactory.TrainTest carTrainTest = fileFactory.getInstancesFromFile(ML.Files.Car, new Options());
        FileFactory.TrainTest censusTrainTest = fileFactory.getInstancesFromFile(ML.Files.Census, new Options());
        return applyCfsSubsetEval(carTrainTest.train) + " \n \n \n \n" + applyCfsSubsetEval(censusTrainTest.train);
    }

    public String ApplyPCA(String name, Instances trainingData) throws Exception{
        AttributeSelection selector = new AttributeSelection();

        PrincipalComponents principalComponents = new PrincipalComponents();
        principalComponents.setMaximumAttributeNames(5);
        principalComponents.setVarianceCovered(.95);
        principalComponents.buildEvaluator(trainingData);

        Ranker ranker = new Ranker();

        selector.setSearch(ranker);
        selector.setEvaluator(principalComponents);
        selector.SelectAttributes(trainingData);
        return name + "\n \n \n \n \n \n \n Principal Components: \n \n "
                + principalComponents.toString() + "\n \n Attribute Selection: \n \n" + selector.toResultsString();
    }

    public Instances applyRP(Instances trainingData, int numAttributes) throws Exception {
        RandomProjection randomProjection = new RandomProjection();
        randomProjection.setNumberOfAttributes(numAttributes);
        randomProjection.setInputFormat(trainingData);
        return Filter.useFilter(trainingData, randomProjection);
    }

    public Instances applyPCAFilter(Instances trainingData, int numAttributes) throws Exception{
        weka.filters.unsupervised.attribute.PrincipalComponents principalComponents = new weka.filters.unsupervised.attribute.PrincipalComponents();
        principalComponents.setMaximumAttributes(numAttributes);
        principalComponents.setInputFormat(trainingData);
        return Filter.useFilter(trainingData, principalComponents);
    }

    public Instances applyICA(Instances trainingData, int numAttributes) throws Exception {
        IndependentComponents independentComponents = new IndependentComponents();
        independentComponents.setNumIterations(100);
        independentComponents.setInputFormat(trainingData);
        return Filter.useFilter(trainingData, independentComponents);
    }

    public String applyCfsSubsetEval(Instances data) throws Exception {
        AttributeSelection selector = new AttributeSelection();
        CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
        cfsSubsetEval.buildEvaluator(data);

        GreedyStepwise greedyStepwise = new GreedyStepwise();
        greedyStepwise.setGenerateRanking(true);
        greedyStepwise.setNumToSelect(4);
        selector.setSearch(greedyStepwise);
        selector.setEvaluator(cfsSubsetEval);
        selector.SelectAttributes(data);

        return  " \n \n Principal Components: \n \n "
                + cfsSubsetEval.toString() + "\n \n Attribute Selection: \n \n" + selector.toResultsString();
    }

    private void plotRP(String fileName, Instances data) throws Exception {
        FileWriter writer = new FileWriter(fileName + ".csv", true);
        for (int i = 0; i < data.size(); i++){
            double[] values = data.get(i).toDoubleArray();
            writer.append(new Double(values[0]).toString());
            writer.append(",");
            writer.append(new Double(values[1]).toString());
            writer.append(",");
            writer.append(new Double(data.get(i).classValue()).toString());
            writer.append("\n");
        }
        writer.flush();
        writer.close();
    }

    public void plotRP() throws Exception{
        FileFactory.TrainTest carTrainTest = fileFactory.getInstancesFromFile(ML.Files.Car, new Options());
        FileFactory.TrainTest censusTrainTest = fileFactory.getInstancesFromFile(ML.Files.Census, new Options());
        plotRP("RP_car", applyRP(carTrainTest.train,2));
        plotRP("RP_census", applyRP(censusTrainTest.train, 2));
    }

    public void plotPCA() throws Exception {
        FileFactory.TrainTest carTrainTest = fileFactory.getInstancesFromFile(ML.Files.Car, new Options());
        FileFactory.TrainTest censusTrainTest = fileFactory.getInstancesFromFile(ML.Files.Census, new Options());
        plotRP("PCA_car", applyPCAFilter(carTrainTest.train, 2));
        plotRP("PCA_census", applyPCAFilter(censusTrainTest.train, 2));
    }

    public void plotICA() throws Exception {
        FileFactory.TrainTest carTrainTest = fileFactory.getInstancesFromFile(ML.Files.CarBin, new Options());
        FileFactory.TrainTest censusTrainTest = fileFactory.getInstancesFromFile(ML.Files.CensusBin, new Options());
        Instances newCarData = applyICA(dropClass(carTrainTest.test), 2);
        Instances newCensusData = applyICA(dropClass(censusTrainTest.test), 2);

        Instances one = reAddClassification(newCarData, carTrainTest.test);
        Instances two = reAddClassification(newCensusData, censusTrainTest.test);

        plotRP("ICA_car", one);
        plotRP("ICA_census", two);
    }

    private Instances dropClass(Instances instances) throws Exception {
        Remove removeFilter = new Remove();
        String[] options = new String[]{"-R", Integer.toString(instances.numAttributes() -1)};
        removeFilter.setOptions(options);
        removeFilter.setInputFormat(instances);
        return Filter.useFilter(instances, removeFilter);
    }

    public Instances reAddClassification(Instances first, Instances second) throws Exception{
        Add filter = new Add();
        filter.setAttributeIndex("last");
        filter.setAttributeName("NewNumeric");
        filter.setInputFormat(first);
        Instances newFirst = Filter.useFilter(first, filter);
        for (int i = 0; i < newFirst.size(); i++){
            newFirst.instance(i).setValue(newFirst.numAttributes() -1, second.instance(i).classValue());
            newFirst.setClassIndex(newFirst.numAttributes() - 1);
        }
        return newFirst;
    }

    public Instances reAddClassificationNominal(Instances first, Instances second) throws Exception{
        Add filter = new Add();
        filter.setAttributeIndex("last");
        filter.setNominalLabels("0,1");
        filter.setAttributeName("NewNumeric");
        filter.setInputFormat(first);
        Instances newFirst = Filter.useFilter(first, filter);

        for (int i = 0; i < newFirst.size(); i++){
            newFirst.instance(i).setValue(newFirst.numAttributes() -1, second.instance(i).classValue());
            newFirst.setClassIndex(newFirst.numAttributes() - 1);
        }
        return newFirst;
    }
}
