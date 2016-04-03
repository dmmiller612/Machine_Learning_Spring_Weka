package com.derek.ml.models;


public class Cluster extends ML{
    private int clusters = 2;
    private Distances distances = Distances.Euclidean;
    private int iterations = 500;

    private FeatureSelection featureSelection = null;

    public int getClusters() {
        return clusters;
    }

    public void setClusters(int clusters) {
        this.clusters = clusters;
    }

    public Distances getDistances() {
        return distances;
    }

    public void setDistances(Distances distances) {
        this.distances = distances;
    }

    public enum Distances {
        Manhatten, Euclidean;
    }

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public void setFileName(Files fileName){
        this.fileName = fileName;
    }

    public Files getFileName(){
        return this.fileName;
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
