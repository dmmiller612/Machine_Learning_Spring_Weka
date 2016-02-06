package com.derek.ml.models;


public class Options {

    public Options(){}

    public Options(boolean featureSelection){
        this.featureSelection = featureSelection;
    }

    private boolean featureSelection = false;

    public boolean isFeatureSelection() {
        return featureSelection;
    }

    public void setFeatureSelection(boolean featureSelection) {
        this.featureSelection = featureSelection;
    }
}
