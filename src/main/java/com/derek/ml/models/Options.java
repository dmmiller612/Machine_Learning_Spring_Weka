package com.derek.ml.models;


public class Options {

    public Options(){}

    public Options(boolean featureSelection){
        this.featureSelection = featureSelection;
    }

    public Options(boolean featureSelection, boolean noClass){
        this.featureSelection = featureSelection;
        this.noClass = noClass;
    }

    private boolean featureSelection = false;
    private boolean noClass = false;

    public boolean isFeatureSelection() {
        return featureSelection;
    }

    public void setFeatureSelection(boolean featureSelection) {
        this.featureSelection = featureSelection;
    }

    public boolean isNoClass() {
        return noClass;
    }

    public void setNoClass(boolean noClass) {
        this.noClass = noClass;
    }

}
