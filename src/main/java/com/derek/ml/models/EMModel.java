package com.derek.ml.models;


public class EMModel extends ML{
    private int iterations = 500;
    private int clusters = 2;
    private double standardDeviations = 1.0E-6D;
    private FeatureSelection featureSelection = null;

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public int getClusters() {
        return clusters;
    }

    public void setClusters(int clusters) {
        this.clusters = clusters;
    }

    public double getStandardDeviations() {
        return standardDeviations;
    }

    public void setStandardDeviations(double standardDeviations) {
        this.standardDeviations = standardDeviations;
    }

    public FeatureSelection getFeatureSelection() {
        return featureSelection;
    }

    public void setFeatureSelection(FeatureSelection featureSelection) {
        this.featureSelection = featureSelection;
    }

    public enum FeatureSelection {
        ICA, PCA, RP, CFS;
    }

}
