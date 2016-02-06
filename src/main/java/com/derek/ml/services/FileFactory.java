package com.derek.ml.services;

import com.derek.ml.models.ML;
import com.derek.ml.models.Options;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.attributeSelection.*;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.*;
import weka.filters.unsupervised.attribute.*;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.util.Map;

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
        }
        return null;
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

    public int[] getFeatureInfo() throws Exception{
        Instances data = handleData("census");
        CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
        cfsSubsetEval.buildEvaluator(data);

        ExhaustiveSearch exhaustiveSearch = new ExhaustiveSearch();
        return exhaustiveSearch.search(cfsSubsetEval, data);
    }

    public TrainTest handlePublicCensus(Options options) throws Exception{
        return handlePublicCensus(0, options);
    }

    public TrainTest handlePublicCensus(int numToRemove, Options options) throws Exception{
        Instances temp = handleData("census");
        Instances trainingData;
        if (numToRemove > 0){
            RemovePercentage removePercentage = new RemovePercentage();
            removePercentage.setInputFormat(temp);
            removePercentage.setPercentage(numToRemove);
            removePercentage.setInvertSelection(true);
            trainingData = Filter.useFilter(temp, removePercentage);
        }else {
            trainingData = temp;
        }
        Instances testData = handleData("censusTest");
        weka.filters.supervised.attribute.Discretize discretize = new weka.filters.supervised.attribute.Discretize();
        discretize.setInputFormat(trainingData);

        Instances trainingDiscretize = Filter.useFilter(trainingData, discretize);
        Instances testingDiscretize = Filter.useFilter(testData, discretize);

        //5,6,8,11,12
        if (options.isFeatureSelection()){
            Instances trainingRemoved = removeFilter(trainingDiscretize, "1-4,7,9-10,13-14");
            Instances testingRemoved = removeFilter(testingDiscretize, "1-4,7,9-10,13-14");
            return new TrainTest(trainingRemoved, testingRemoved);
        }
        return new TrainTest(trainingDiscretize, testingDiscretize);
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


    public class TrainTest {
        public Instances train;
        public Instances test;
        public TrainTest(Instances train, Instances test){
            this.train = train;
            this.test = test;
        }
    }
}
