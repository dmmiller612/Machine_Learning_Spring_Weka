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
        switch (file){
            case Census:
                return handlePublicCensus(options);
            case Car:
                return new TrainTest(handleData("car_train"), handleData("car_test"));
            case Mammograph:
                Instances data = removeInstancesWithQuestionMarks(handleData("mammograph"));
                Instances removedData = removeFilter(data, "1");
                Instances numericToNominal = numericToNominalFilter(removedData, "2-5");
                Instances fin = discretizeFilter(numericToNominal, "1", 4);

                CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
                cfsSubsetEval.buildEvaluator(fin);
                ExhaustiveSearch exhaustiveSearch = new ExhaustiveSearch();
                int[] is = exhaustiveSearch.search(cfsSubsetEval, fin);
                for (int x : is){
                    System.out.println(x);
                }
                return new TrainTest(fin, null);
            case Stock:
                Instances stockData = handleData("stock");
                stockData.setClassIndex(stockData.numAttributes()-1);
                return new TrainTest(stockData, null);
            case Nba:
                Instances nbaData = handleData("nba");
                nbaData.setClassIndex(nbaData.numAttributes()-1);
                return new TrainTest(nbaData, null);
            case Connect4:
                Instances connect4 = handleData("connect4");
                connect4.setClassIndex(connect4.numAttributes()-1);
                return new TrainTest(connect4, handleData("connect4Test"));
            case Mushrooms:
                Instances mushrooms = handleData("mushroom");
                mushrooms.setClassIndex(0);
                return new TrainTest(mushrooms, null);
            case Wine:
                Instances wine = handleData("wine");
                Discretize dis1 = new Discretize();
                dis1.setIgnoreClass(true);
                dis1.setBins(5);
                dis1.setAttributeIndices("1-11");
                dis1.setInputFormat(wine);

                Instances in = Filter.useFilter(wine, dis1);
                Discretize dis2 = new Discretize();
                dis2.setIgnoreClass(true);
                dis2.setBins(2);
                dis2.setAttributeIndices("12");
                dis2.setInputFormat(in);
                return new TrainTest(Filter.useFilter(in, dis2), null);
            case Seismic:
                return new TrainTest(handleData("seismic"),null);
            case Contraceptive:
                Instances contraceptive = handleData("contraceptive");
                Instances ntn = numericToNominalFilter(contraceptive, "2-3, 5-9, 10");
                ntn.setClassIndex(ntn.numAttributes() -1);
                return new TrainTest(discretizeFilter(ntn, "1,4,10", 4), null);
            case TicTac:
                return new TrainTest(handleData("tictac"), handleData("tictacTest"));
            case Bank:
                Instances bank = handleData("bank");
                //discretizeFilter(handleData("Bank"), "1,6,11-14", 6);
                weka.filters.supervised.attribute.Discretize dis = new weka.filters.supervised.attribute.Discretize();
                dis.setInputFormat(bank);
                return new TrainTest(Filter.useFilter(bank, dis), null);
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
