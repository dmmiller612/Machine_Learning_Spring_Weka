package com.derek.ml.models;


public class NearestNeighbor extends ML {

    private TreeTypes treeTypes = TreeTypes.Linear;
    private int k = 1;
    private boolean holdOneOut = false;
    private boolean useMeanError = false;
    private boolean boost = false;
    private boolean featureSelection = false;

    public NearestNeighbor(){}

    public NearestNeighbor(int k, ML.Files fileName){
        this.k = k;
        super.setFileName(fileName);
    }

    public NearestNeighbor(int k, ML.Files fileName, boolean featureSelection){
        this.k = k;
        super.setFileName(fileName);
        this.featureSelection = featureSelection;
    }

    public static enum TreeTypes {
        BallTree, CoverTree, Linear
    }

    public TreeTypes getTreeTypes() {
        return treeTypes;
    }

    public void setTreeTypes(TreeTypes treeTypes) {
        this.treeTypes = treeTypes;
    }

    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
    }

    public boolean isHoldOneOut() {
        return holdOneOut;
    }

    public void setHoldOneOut(boolean holdOneOut) {
        this.holdOneOut = holdOneOut;
    }

    public boolean isUseMeanError() {
        return useMeanError;
    }

    public void setUseMeanError(boolean useMeanError) {
        this.useMeanError = useMeanError;
    }

    public void setFileName(Files fileName){
        super.setFileName(fileName);
    }

    public Files getFileName(){
        return super.getFileName();
    }

    public void setTestType(TestType testType){
        super.setTestType(testType);
    }

    public TestType getTestType(){
        return super.getTestType();
    }

    public boolean isBoost() {
        return boost;
    }

    public void setBoost(boolean boost) {
        this.boost = boost;
    }

    public boolean isFeatureSelection() {
        return featureSelection;
    }

    public void setFeatureSelection(boolean featureSelection) {
        this.featureSelection = featureSelection;
    }


}
