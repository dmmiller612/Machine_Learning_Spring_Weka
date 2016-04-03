package com.derek.ml.services;

import com.derek.ml.models.ML;
import com.derek.ml.models.Options;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.attributeSelection.*;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.instance.RemovePercentage;

@Service
public class FileFactory {

    @Autowired
    public LoadData loadData;

    public TrainTest getInstancesFromFile(ML.Files file, Options options) throws Exception{
        switch (file) {
            case Census:
                return handlePublicCensus(options);
            case Car:
                return new TrainTest(handleData("car_train"), handleData("car_test"));
            case Boston:
                return new TrainTest(handleData("boston"), handleData("boston"));
            case CarBin:
                return new TrainTest(handleData("car_bin", true), handleData("car_bin"));
            case CensusBin:
                return new TrainTest(handleData("census_bin", true), handleData("census_bin"));
            case CensusEm:
                return new TrainTest(handleData("census_em"), handleData("census_em"));
            case CensusKm:
                return new TrainTest(handleData("census_km"), handleData("census_km"));
        }
        return null;
    }

    private Instances filterClass(Instances data) throws Exception{
        Remove filter = new Remove();
        filter.setAttributeIndices("" + (data.classIndex() + 1));
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }

    public void splitCarDataToTest(int amount) throws Exception{
        Instances instances = handleData("car");
        RemovePercentage removePercentage = new RemovePercentage();
        removePercentage.setPercentage(amount);
        removePercentage.setInputFormat(instances);
        loadData.saveToArff(Filter.useFilter(instances, removePercentage), "car_train.arff");
        removePercentage.setInvertSelection(true);
        loadData.saveToArff(Filter.useFilter(instances, removePercentage), "car_test.arff");
    }

    private Instances handleData(String fileName) throws Exception{
        try {
            return loadData.getDataFromArff(fileName + ".arff");
        } catch (Exception e){
            return loadData.getDataFromCsvFile(fileName + ".csv");
        }
    }

    private Instances handleData(String fileName, boolean noClass) throws Exception{
        try {
            return loadData.getDataFromArff(fileName + ".arff", noClass);
        } catch (Exception e){
            return loadData.getDataFromCsvFile(fileName + ".csv");
        }
    }

    private Instances removeFilter(Instances data, String indicesToRemove) throws Exception{
        Remove remove = new Remove();
        remove.setAttributeIndices(indicesToRemove);
        remove.setInputFormat(data);
        remove.setInvertSelection(true);
        return Filter.useFilter(data, remove);
    }

    private Instances numericToNominalFilter(Instances data, String indicesToNominalize) throws Exception {
        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setAttributeIndices(indicesToNominalize);
        numericToNominal.setInputFormat(data);
        return Filter.useFilter(data, numericToNominal);
    }

    private Instances discretizeFilter(Instances data, String indices, int bins) throws Exception{
        Discretize d = new Discretize();
        if (indices != null){
            d.setAttributeIndices(indices);
        }
        d.setIgnoreClass(true);
        d.setBins(bins);
        d.setInputFormat(data);
        return Filter.useFilter(data, d);
    }

    private Instances removeInstancesWithQuestionMarks(Instances data){
        int numAttributes = data.numAttributes();
        int numInstances = data.numInstances();

        for (int out = 0; out < numInstances; out++){
            Instance currentInstance = data.instance(out);
            for (int in = 0; in < numAttributes; in++){
                if (currentInstance != null){
                    try{
                        String currentAttribute = currentInstance.toString(in);
                        if (currentAttribute.contains("?")){
                            data.delete(out);
                        }
                    } catch (Exception e){
                        //
                    }
                }
            }
        }
        return data;
    }

    public TrainTest handlePublicCensus(Options options) throws Exception{
        return handlePublicCensus(0, options);
    }

    public TrainTest handlePublicCensus(int numToRemove, Options options) throws Exception{
        Instances trainingData = handleData("census", options.isNoClass());
        Instances testData;
        if (options.isNoClass()){
            testData = handleData("census", false);
        } else {
            testData = handleData("censusTest", false);
        }

        Instances temp;
        if (numToRemove > 0){
            RemovePercentage removePercentage = new RemovePercentage();
            removePercentage.setInputFormat(trainingData);
            removePercentage.setPercentage(numToRemove);
            removePercentage.setInvertSelection(true);
            temp = Filter.useFilter(trainingData, removePercentage);
        }else {
            temp = trainingData;
        }

        //5,6,8,11,12
        if (options.isFeatureSelection()){
            Instances trainingRemoved = removeFilter(temp, "1-4,7,9-10,13-14");
            Instances testingRemoved = removeFilter(testData, "1-4,7,9-10,13-14");
            return new TrainTest(trainingRemoved, testingRemoved);
        }

        return new TrainTest(temp, testData);
    }

    public TrainTest handlePublicCar(int num) throws Exception{
        TrainTest data = new TrainTest(handleData("car_train"), handleData("car_test"));
        if (num <= 0){
            return data;
        }
        RemovePercentage removePercentage = new RemovePercentage();
        removePercentage.setInputFormat(data.train);
        removePercentage.setPercentage(num);
        removePercentage.setInvertSelection(true);
        Instances trainingData = Filter.useFilter(data.train, removePercentage);

        return new TrainTest(trainingData, data.test);
    }

    public void saveDiscretizedArff() throws Exception{
        Instances temp = handleData("census");
        Instances testData = handleData("censusTest");
        weka.filters.supervised.attribute.Discretize discretize = new weka.filters.supervised.attribute.Discretize();
        discretize.setInputFormat(temp);

        Instances trainingDiscretize = Filter.useFilter(temp, discretize);
        Instances testingDiscretize = Filter.useFilter(testData, discretize);
        loadData.saveToArff(trainingDiscretize, "census.arff");
        loadData.saveToArff(testingDiscretize, "censusTest.arff");
    }

    public class TrainTest {
        public Instances train;
        public Instances test;
        public TrainTest(Instances train, Instances test){
            this.train = train;
            this.test = test;
        }
    }
}
