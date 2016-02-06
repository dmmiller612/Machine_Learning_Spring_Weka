package com.derek.ml.services;

import com.derek.ml.models.ML;
import com.derek.ml.models.Options;
import com.derek.ml.models.SVMModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.SelectedTag;

@Service
public class SVMService {

    @Autowired
    private FileFactory fileFactory;

    @Autowired
    private EvaluationService evaluationService;

    @Autowired
    LoadData loadData;

    public String handleSVM(SVMModel svmModel) throws Exception{
        FileFactory.TrainTest instances = fileFactory.getInstancesFromFile(svmModel.getFileName(), new Options(svmModel.isFeatureSelection()));
        LibSVM libSVM = new LibSVM();
        if (svmModel.getKernelType() == SVMModel.KernelType.Sigmoid){
            libSVM.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_SIGMOID, LibSVM.TAGS_KERNELTYPE));
        }
        else if (svmModel.getKernelType() == SVMModel.KernelType.Linear){
            libSVM.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
        }
        else if (svmModel.getKernelType() == SVMModel.KernelType.Polynomial){
            libSVM.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_POLYNOMIAL, LibSVM.TAGS_KERNELTYPE));
        }
        else if (svmModel.getKernelType() == SVMModel.KernelType.RBF){
            libSVM.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_RBF, LibSVM.TAGS_KERNELTYPE));
        }
        libSVM.buildClassifier(instances.train);
        return handleLibSvmEvaluation(libSVM, svmModel, instances);
    }

    public String handleSplitData(SVMModel svmModel, int num, String retString) throws Exception{
        if (num <= 100){
            retString += "Amount " + Integer.toString(num) + "\n";
            FileFactory.TrainTest data;
            if (svmModel.getFileName() == ML.Files.Census){
                data = fileFactory.handlePublicCensus(num, new Options(svmModel.isFeatureSelection()));
            } else {
                data = fileFactory.handlePublicCar(num);
            }

            LibSVM cls = svmClassifier(svmModel, data.train);
            return handleSplitData(svmModel, num==1 ? num+9 : num+10, retString + "\n \n" + evaluationService.evaluateData(data.train, cls, data.test));
        }
        return retString;
    }

    public void createModel(SVMModel svm) throws Exception{
        FileFactory.TrainTest data = fileFactory.getInstancesFromFile(svm.getFileName(), new Options(svm.isFeatureSelection()));
        Classifier cls = svmClassifier(svm, data.train);
        loadData.saveModel(cls, getString(svm));
    }

    public String getModel(SVMModel svm) throws Exception {
        FileFactory.TrainTest data = fileFactory.getInstancesFromFile(svm.getFileName(), new Options(svm.isFeatureSelection()));
        Classifier cls = loadData.getModel(getString(svm));
        svm.setTestType(ML.TestType.TestData);
        return handleLibSvmEvaluation((LibSVM) cls, svm, data);
    }

    private String getString(SVMModel svm){
        String s = "SVM-KernelType=" + svm.getKernelType().toString() +
                "-FileName=" + svm.getFileName();
        if (svm.isFeatureSelection()){
            s += "-feature=true";
        }
        return s + ".model";
    }

    private LibSVM svmClassifier(SVMModel svmModel, Instances data) throws Exception{
        LibSVM libSVM = new LibSVM();
        if (svmModel.getKernelType() == SVMModel.KernelType.Sigmoid){
            libSVM.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_SIGMOID, LibSVM.TAGS_KERNELTYPE));
        }
        else if (svmModel.getKernelType() == SVMModel.KernelType.Linear){
            libSVM.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
        }
        else if (svmModel.getKernelType() == SVMModel.KernelType.Polynomial){
            libSVM.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_POLYNOMIAL, LibSVM.TAGS_KERNELTYPE));
        }
        else if (svmModel.getKernelType() == SVMModel.KernelType.RBF){
            libSVM.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_RBF, LibSVM.TAGS_KERNELTYPE));
        }
        libSVM.buildClassifier(data);
        return libSVM;
    }

    private String handleLibSvmEvaluation(LibSVM libSVM, SVMModel svmModel, FileFactory.TrainTest instances) throws Exception{
        if (svmModel.getTestType().equals(ML.TestType.CrossValidation)){
            return evaluationService.evaluateData(instances.train, libSVM, 10);
        }
        else if (svmModel.getTestType().equals(ML.TestType.Train)){
            return evaluationService.evaluateData(instances.train, libSVM, instances.train);
        }
        else {
            return evaluationService.evaluateData(instances.train, libSVM, instances.test);
        }
    }

}
