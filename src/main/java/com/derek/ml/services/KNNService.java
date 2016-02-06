package com.derek.ml.services;

import com.derek.ml.models.ML;
import com.derek.ml.models.NearestNeighbor;
import com.derek.ml.models.Options;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;
import weka.core.neighboursearch.BallTree;
import weka.core.neighboursearch.CoverTree;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.util.Enumeration;
import java.util.Random;

@Service
public class KNNService {

    @Autowired
    private FileFactory fileFactory;

    @Autowired
    private EvaluationService evaluationService;

    @Autowired
    LoadData loadData;

    public String handleKNNService(NearestNeighbor nearestNeighbor) throws Exception{
        FileFactory.TrainTest instances = fileFactory.getInstancesFromFile(nearestNeighbor.getFileName(), new Options(nearestNeighbor.isFeatureSelection()));
        Classifier iBk = handleIBK(nearestNeighbor, instances.train);
        return evaluateKNN(nearestNeighbor, instances, iBk);
    }

    public Classifier handleIBK(NearestNeighbor nearestNeighbor, Instances instances) throws Exception{
        IBk iBk = new IBk();
        iBk.setKNN(nearestNeighbor.getK());
        iBk.setNearestNeighbourSearchAlgorithm(getNearestNeighborAlgorithm(nearestNeighbor));
        iBk.setCrossValidate(nearestNeighbor.isHoldOneOut());
        iBk.setMeanSquared(nearestNeighbor.isUseMeanError());
        if (nearestNeighbor.isBoost()){
            AdaBoostM1 adaBoostM1 = new AdaBoostM1();
            adaBoostM1.setClassifier(iBk);
            adaBoostM1.buildClassifier(instances);
            return adaBoostM1;
        }
        iBk.buildClassifier(instances);
        return iBk;
    }

    public String evaluateKNN(NearestNeighbor nearestNeighbor, FileFactory.TrainTest instances, Classifier iBk) throws Exception{
        if (nearestNeighbor.getTestType() == ML.TestType.TestData){
            return evaluationService.evaluateData(instances.train, iBk, instances.test)  + "\n \n " + iBk.toString();
        } else if (nearestNeighbor.getTestType() == ML.TestType.Train){
            return evaluationService.evaluateData(instances.train, iBk, instances.train) + "\n \n " + iBk.toString();
        }
        return evaluationService.evaluateData(instances.train, iBk) + "\n \n " + iBk.toString();
    }

    public void createModel(NearestNeighbor nearestNeighbor) throws Exception{
        FileFactory.TrainTest data = fileFactory.getInstancesFromFile(nearestNeighbor.getFileName(), new Options(nearestNeighbor.isFeatureSelection()));
        Classifier cls = handleIBK(nearestNeighbor, data.train);
        loadData.saveModel(cls, getString(nearestNeighbor));
    }

    public String getModel(NearestNeighbor nearestNeighbor) throws Exception{
        FileFactory.TrainTest data = fileFactory.getInstancesFromFile(nearestNeighbor.getFileName(), new Options(nearestNeighbor.isFeatureSelection()));
        Classifier cls = loadData.getModel(getString(nearestNeighbor));
        nearestNeighbor.setTestType(ML.TestType.TestData);
        return evaluateKNN(nearestNeighbor, data, cls);
    }

    private String getString(NearestNeighbor nearestNeighbor){
        String nn = "KNearestNeighbor-k=" + nearestNeighbor.getK() + "-fileName=" + nearestNeighbor.getFileName();
        if (nearestNeighbor.isFeatureSelection()){
            nn += "-feature=true";
        }
        return nn + ".model";
    }

    public String handleSplitData(NearestNeighbor nearestNeighbor, int num, String retString) throws Exception{
        if (num <= 100){
            retString += "Amount " + Integer.toString(num) + "\n";
            FileFactory.TrainTest data;
            if (nearestNeighbor.getFileName() == ML.Files.Census){
                data = fileFactory.handlePublicCensus(num, new Options(nearestNeighbor.isFeatureSelection()));
            } else {
                data = fileFactory.handlePublicCar(num);
            }
            Classifier cls = handleIBK(nearestNeighbor, data.train);
            return handleSplitData(nearestNeighbor, num==1 ? num+9 : num+10, retString + "\n \n" + evaluationService.evaluateData(data.train, cls, data.test));
        }
        return retString;
    }

    private NearestNeighbourSearch getNearestNeighborAlgorithm(NearestNeighbor nearestNeighbor){
        if (nearestNeighbor.getTreeTypes() == NearestNeighbor.TreeTypes.Linear){
            return new LinearNNSearch();
        } else if (nearestNeighbor.getTreeTypes() == NearestNeighbor.TreeTypes.BallTree){
            return new BallTree();
        } else if (nearestNeighbor.getTreeTypes() == NearestNeighbor.TreeTypes.CoverTree){
            return new CoverTree();
        }
        return null;
    }

}
