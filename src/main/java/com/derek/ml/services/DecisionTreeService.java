package com.derek.ml.services;

import com.derek.ml.models.DecisionTree;
import com.derek.ml.models.ML;
import com.derek.ml.models.Options;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;

import java.util.Random;

@Service
public class DecisionTreeService {

    private LoadData loadData;
    private FileFactory fileFactory;
    private EvaluationService evaluationService;

    @Autowired
    public DecisionTreeService(FileFactory fileFactory, LoadData loadData, EvaluationService evaluationService){
        this.fileFactory = fileFactory;
        this.loadData = loadData;
        this.evaluationService = evaluationService;
    }

    public String getDecisionTreeInformation(DecisionTree decisionTree) throws Exception{
        if (decisionTree.getTreeType() == DecisionTree.TreeType.ID3){
            return handleId3(decisionTree);
        } else if (decisionTree.getTreeType() == DecisionTree.TreeType.J48) {
            return handleJ48(decisionTree);
        }
        return null;
    }

    public Classifier buildJ48(DecisionTree decisionTree, Instances data) throws Exception{
        //uses information gain ratio
        J48 j48 = new J48();
        if (decisionTree.getUnpruned() != null){
            j48.setUnpruned(decisionTree.getUnpruned());
        }
        if (decisionTree.getConfidence() != null){
            j48.setOptions(new String[]{"-C", decisionTree.getConfidence()});
        }
        if (decisionTree.getMinNumObj() != null){
            j48.setMinNumObj(decisionTree.getMinNumObj());
        }
        if (decisionTree.isBoost()){
            AdaBoostM1 adaBoostM1 = new AdaBoostM1();
            adaBoostM1.setUseResampling(false);
            adaBoostM1.setClassifier(j48);
            adaBoostM1.buildClassifier(data);
            return adaBoostM1;
        } else {
            j48.buildClassifier(data);
            return j48;
        }
    }

    public String evaluateData(FileFactory.TrainTest data, Classifier classifier, DecisionTree decisionTree) throws Exception{
        Evaluation evaluation = new Evaluation(data.train);
        if (decisionTree.getTestType() == DecisionTree.TestType.CrossValidation){
            evaluation.crossValidateModel(classifier, data.train, 10, new Random(1));
        } else if (decisionTree.getTestType() == DecisionTree.TestType.Train){
            FileFactory.TrainTest d;
            if (decisionTree.getFileName() == ML.Files.Census){
                d = fileFactory.handlePublicCensus(0, new Options(decisionTree.isFeatureSelection()));
            } else {
                d = fileFactory.handlePublicCar(0);
            }
            evaluation.evaluateModel(classifier, d.train);
        } else {
            evaluation.evaluateModel(classifier, data.test);
        }
        String retString = "";
        retString += evaluation.toSummaryString() + " \n";
        retString += evaluation.toClassDetailsString() + " \n";
        retString += evaluation.toMatrixString() + " \n";
        return retString;
    }

    public void createModel(DecisionTree decisionTree) throws Exception{
        FileFactory.TrainTest data = fileFactory.getInstancesFromFile(decisionTree.getFileName(), new Options(decisionTree.isFeatureSelection()));
        Classifier cls = buildJ48(decisionTree, data.train);
        loadData.saveModel(cls, getString(decisionTree));
    }

    public String getModel(DecisionTree decisionTree) throws Exception{
        FileFactory.TrainTest data = fileFactory.getInstancesFromFile(decisionTree.getFileName(), new Options(decisionTree.isFeatureSelection()));
        Classifier cls = loadData.getModel(getString(decisionTree));
        decisionTree.setTestType(ML.TestType.TestData);
        return evaluateData(data, cls, decisionTree);
    }

    private String getString(DecisionTree decisionTree){
        String dString = "decisionTree-minNumObj=" +
                decisionTree.getMinNumObj().toString() +
                "-Boosted=" + decisionTree.isBoost() +
                "-C=" + decisionTree.getConfidence() +
                "-file=" + decisionTree.getFileName().toString();
        if (decisionTree.isFeatureSelection()){
            dString += "-feature=true";
        }
        return dString + ".model";
    }

    public String handleSplitData(DecisionTree decisionTree, int num, String retString) throws Exception{
        if (num <= 100){
            retString += "Amount " + Integer.toString(num) + "\n";
            FileFactory.TrainTest data;
            if (decisionTree.getFileName() == ML.Files.Census){
                data = fileFactory.handlePublicCensus(num, new Options(decisionTree.isFeatureSelection()));
            } else {
                data = fileFactory.handlePublicCar(num);
            }
            Classifier cls = buildJ48(decisionTree, data.train);
            Instances d;
            if (decisionTree.getTestType() == ML.TestType.Train){
                if (decisionTree.getFileName() == ML.Files.Car){
                    d = fileFactory.handlePublicCar(0).train;
                } else {
                    d = fileFactory.handlePublicCensus(0, new Options(decisionTree.isFeatureSelection())).train;
                }
            } else {
                d = data.test;
            }
            return handleSplitData(decisionTree, num==1 ? num+9 : num+10, retString + "\n \n" + evaluationService.evaluateData(data.train, cls, d));
        }
        return retString;
    }

    private String handleId3(DecisionTree decisionTree) throws Exception{
        FileFactory.TrainTest data = fileFactory.getInstancesFromFile(decisionTree.getFileName(), new Options(decisionTree.isFeatureSelection()));
        Id3 id3 = buildId3(data.train);
        return evaluateData(data, id3, decisionTree);
    }

    private Id3 buildId3(Instances data) throws Exception{
        Id3 id3 = new Id3();
        id3.buildClassifier(data);
        return id3;
    }

    private String handleJ48(DecisionTree decisionTree) throws Exception{
        FileFactory.TrainTest data = fileFactory.getInstancesFromFile(decisionTree.getFileName(), new Options(decisionTree.isFeatureSelection()));
        Classifier j48 = buildJ48(decisionTree, data.train);
        return evaluateData(data, j48, decisionTree) + "\n \n " + j48.toString();
    }

}
