package com.derek.ml.services;

import com.derek.ml.models.ML;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

@Service
public class EvaluationService {

    @Autowired
    FileFactory fileFactory;

    public String evaluateData(Instances data, Classifier classifier) throws Exception{
        return evaluateData(data, classifier, 10);
    }

    public String evaluateData(Instances data, Classifier classifier, int numberOfFolds) throws Exception{
        return evaluateData(data, classifier, numberOfFolds, null);
    }

    public String evaluateData(Instances data, Classifier classifier, Instances testData) throws Exception {
        return evaluateData(data, classifier, 10, testData);
    }

    public String evaluateData(Instances data, Classifier classifier, int numberOfFolds, Instances testData) throws Exception{
        Evaluation evaluation = new Evaluation(data);
        if (testData == null){
            evaluation.crossValidateModel(classifier, data, numberOfFolds, new Random(1));
        } else {
            evaluation.evaluateModel(classifier, testData);
        }
        String retString = "";
        retString += evaluation.toSummaryString() + " \n";
        retString += evaluation.toClassDetailsString() + " \n";
        retString += evaluation.toMatrixString() + " \n";
        return retString;
    }
}
