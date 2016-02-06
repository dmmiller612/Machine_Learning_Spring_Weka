package com.derek.ml.models;


public class SVMModel extends ML {

    private KernelType kernelType = KernelType.Linear;
    private boolean featureSelection = false;

    public SVMModel(){}

    public SVMModel(KernelType kernelType, ML.Files fileName){
        this.kernelType = kernelType;
        super.setFileName(fileName);
    }

    public static enum KernelType {
        Linear, Sigmoid, Polynomial, RBF;
    }

    public void setFileName(Files fileName){
        super.setFileName(fileName);
    }
    public Files getFileName(){
        return super.getFileName();
    }

    public KernelType getKernelType() {
        return kernelType;
    }

    public void setKernelType(KernelType kernelType) {
        this.kernelType = kernelType;
    }

    public void setTestType(TestType testType){
        super.setTestType(testType);
    }

    public TestType getTestType(){
        return super.getTestType();
    }

    public boolean isFeatureSelection() {
        return featureSelection;
    }

    public void setFeatureSelection(boolean featureSelection) {
        this.featureSelection = featureSelection;
    }
}
