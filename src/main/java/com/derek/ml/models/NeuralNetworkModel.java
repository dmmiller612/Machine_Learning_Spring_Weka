package com.derek.ml.models;


public class NeuralNetworkModel extends ML{
    private int epochRate = 250;
    private int hiddenLayers = 1;
    private boolean featureSelection = false;

    public NeuralNetworkModel(){}

    public NeuralNetworkModel(int hiddenLayers, int epochRate, ML.Files fileName){
        this.hiddenLayers = hiddenLayers;
        this.epochRate = epochRate;
        super.setFileName(fileName);
    }

    public int getEpochRate() {
        return epochRate;
    }

    public void setEpochRate(int epochRate) {
        this.epochRate = epochRate;
    }

    public int getHiddenLayers() {
        return hiddenLayers;
    }

    public void setHiddenLayers(int hiddenLayers) {
        this.hiddenLayers = hiddenLayers;
    }

    public void setFileName(Files fileName){
        super.setFileName(fileName);
    }

    public Files getFileName(){
        return super.getFileName();
    }

    public boolean isFeatureSelection() {
        return featureSelection;
    }

    public void setFeatureSelection(boolean featureSelection) {
        this.featureSelection = featureSelection;
    }
}
