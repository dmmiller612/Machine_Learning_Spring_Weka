package com.derek.ml.models;


public class FeatureReduction {
    private double varianceCovered = .95;
    private int maximumAttributeNames = 10;
    private double percent = 50;
    private int numberOfAttributes = 7;
    private int numberOfIterations = 100;


    public int getNumberOfIterations() {
        return numberOfIterations;
    }

    public void setNumberOfIterations(int numberOfIterations) {
        this.numberOfIterations = numberOfIterations;
    }

    public double getVarianceCovered() {
        return varianceCovered;
    }

    public void setVarianceCovered(double varianceCovered) {
        this.varianceCovered = varianceCovered;
    }

    public int getMaximumAttributeNames() {
        return maximumAttributeNames;
    }

    public void setMaximumAttributeNames(int maximumAttributeNames) {
        this.maximumAttributeNames = maximumAttributeNames;
    }

    public double getPercent() {
        return percent;
    }

    public void setPercent(double percent) {
        this.percent = percent;
    }

    public int getNumberOfAttributes() {
        return numberOfAttributes;
    }

    public void setNumberOfAttributes(int numberOfAttributes) {
        this.numberOfAttributes = numberOfAttributes;
    }

}
