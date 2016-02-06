package com.derek.ml.models;


public class DecisionTree extends ML{

    private TreeType treeType = TreeType.J48;
    private String confidence = "0.25";
    private Boolean unpruned = false;
    private Integer minNumObj = 2;
    private boolean boost = false;
    private boolean featureSelection = false;

    public DecisionTree(){}

    public DecisionTree(int minNumObj, String confidence, boolean boost, ML.Files fileName){
        this.minNumObj = minNumObj;
        this.confidence = confidence;
        this.boost = boost;
        super.setFileName(fileName);
    }

    public static enum TreeType {
        ID3,
        J48,
        ALL
    }

    public Integer getMinNumObj() {
        return minNumObj;
    }

    public void setMinNumObj(Integer minNumObj) {
        this.minNumObj = minNumObj;
    }

    public TreeType getTreeType() {
        return treeType;
    }

    public void setTreeType(TreeType treeType) {
        this.treeType = treeType;
    }

    public String getConfidence() {
        return confidence;
    }

    public void setConfidence(String confidence) {
        this.confidence = confidence;
    }

    public Boolean getUnpruned() {
        return unpruned;
    }

    public void setUnpruned(Boolean unpruned) {
        this.unpruned = unpruned;
    }

    public boolean isBoost() {
        return boost;
    }

    public void setBoost(boolean boost) {
        this.boost = boost;
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
