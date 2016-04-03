package com.derek.ml.services;

import com.derek.ml.models.EMModel;
import com.derek.ml.models.Cluster;
import com.derek.ml.models.ML;
import com.derek.ml.models.Options;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.MakeDensityBasedClusterer;
import weka.clusterers.SimpleKMeans;
import weka.clusterers.EM;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileWriter;

@Service
public class ClusterService {

    @Autowired
    private FileFactory fileFactory;

    @Autowired
    private EvaluationService evaluationService;

    @Autowired
    FeatureReductionService featureReductionService;

    public String handleKmeans(Cluster cluster) throws Exception{
        FileFactory.TrainTest trainTest = fileFactory.getInstancesFromFile(cluster.getFileName(), new Options(false, true));
        Instances data = trainTest.train;
        if (cluster.getFeatureSelection() != null){
            data = applyFeatureSelection(trainTest, cluster);
        }
        MakeDensityBasedClusterer simpleKMeans = makeDensityBasedClustererWrapper(trainKmeans(cluster, data), data);

        ClusterEvaluation clusterEvaluation = new ClusterEvaluation();
        clusterEvaluation.setClusterer(simpleKMeans);
        Instances instancesToEvaluate = trainTest.test;
        if (cluster.getFeatureSelection() != null){
            instancesToEvaluate = featureReductionService.reAddClassificationNominal(data, trainTest.test);
        }
        clusterEvaluation.evaluateClusterer(instancesToEvaluate);
        return clusterEvaluation.clusterResultsToString() + "\n \n \n Log Likelihood : " + clusterEvaluation.getLogLikelihood();
    }

    public SimpleKMeans trainKmeans(Cluster cluster, Instances data) throws Exception{
        SimpleKMeans simpleKMeans = new SimpleKMeans();
        simpleKMeans.setPreserveInstancesOrder(true);
        simpleKMeans.setNumClusters(cluster.getClusters());
        if (cluster.getDistances() == Cluster.Distances.Euclidean){
            simpleKMeans.setDistanceFunction(new EuclideanDistance());
        } else {
            simpleKMeans.setDistanceFunction(new ManhattanDistance());
        }
        simpleKMeans.setMaxIterations(cluster.getIterations());
        simpleKMeans.buildClusterer(data);
        return simpleKMeans;
    }

    public String handleEM(Cluster emModel) throws Exception{
        FileFactory.TrainTest trainTest = fileFactory.getInstancesFromFile(emModel.getFileName(), new Options(false, true));
        Instances data = trainTest.train;
        if (emModel.getFeatureSelection() != null){
            data = applyFeatureSelection(trainTest, emModel);
        }
        EM em = trainEm(emModel, data);

        ClusterEvaluation clusterEvaluation = new ClusterEvaluation();
        clusterEvaluation.setClusterer(em);
        clusterEvaluation.evaluateClusterer(featureReductionService.reAddClassificationNominal(data, trainTest.test));

        return clusterEvaluation.clusterResultsToString();
    }

    public EM trainEm(Cluster emModel, Instances data) throws Exception{
        EM em = new EM();
        em.setMaxIterations(emModel.getIterations());
        em.setNumClusters(emModel.getClusters());
        em.buildClusterer(data);
        return em;
    }

    public void plotKM(Cluster cluster, Instances instances, String name) throws Exception{
        SimpleKMeans simpleKMeans = trainKmeans(cluster, instances);
        int[] assignments = simpleKMeans.getAssignments();

        try{
            FileWriter writer = new FileWriter(name + ".csv", true);
            for (int i = 0; i < assignments.length; i++){
                double[] values = instances.get(i).toDoubleArray();
                writer.append(new Double(values[0]).toString());
                writer.append(",");
                writer.append(new Double(values[1]).toString());
                writer.append(",");
                writer.append(new Integer(assignments[i]).toString());
                writer.append("\n");
            }
            writer.flush();
            writer.close();

        } catch (Exception e){
            System.out.println(e.toString());
        }
    }

    public void plotEM(Cluster emModel, Instances instances, String name) throws Exception{
        EM simpleEM = trainEm(emModel, instances);
        ClusterEvaluation clusterEvaluation = new ClusterEvaluation();
        clusterEvaluation.setClusterer(simpleEM);
        clusterEvaluation.evaluateClusterer(instances);
        double[] assignments = clusterEvaluation.getClusterAssignments();

        try{
            FileWriter writer = new FileWriter(name + ".csv", true);
            for (int i = 0; i < assignments.length; i++){
                double[] values = instances.get(i).toDoubleArray();
                writer.append(new Double(values[0]).toString());
                writer.append(",");
                writer.append(new Double(values[1]).toString());
                writer.append(",");
                writer.append(new Double(assignments[i]).toString());
                writer.append("\n");
            }
            writer.flush();
            writer.close();

        } catch (Exception e){
            System.out.println(e.toString());
        }
    }

    public void plotKMWithFeature() throws Exception{
        FileFactory.TrainTest carTrainTest = fileFactory.getInstancesFromFile(ML.Files.CarBin, new Options(false, true));
        FileFactory.TrainTest censusTrainTest = fileFactory.getInstancesFromFile(ML.Files.CensusBin, new Options(false, true));
        FileFactory.TrainTest carBin = fileFactory.getInstancesFromFile(ML.Files.CarBin, new Options(false, true));
        FileFactory.TrainTest censusBin = fileFactory.getInstancesFromFile(ML.Files.CensusBin, new Options(false, true));

        Instances pcaCar = featureReductionService.applyPCAFilter(carTrainTest.train, 30);
        Instances pcaCensus = featureReductionService.applyPCAFilter(censusTrainTest.train, 30);
        Instances icaCar = featureReductionService.applyICA(carBin.test, 30);
        Instances icaCensus = featureReductionService.applyICA(censusBin.test, 30);
        Instances rpCar = featureReductionService.applyRP(carBin.train, 30);
        Instances rpCensus = featureReductionService.applyRP(censusBin.train, 30);

        Cluster cluster = new Cluster();
        cluster.setClusters(6);
        cluster.setIterations(1000);

        plotKM(cluster, pcaCar, "PCA_CAR");
        plotKM(cluster, pcaCensus, "PCA_CENSUS");
        plotKM(cluster, filterClass(icaCar), "ICA_CAR");
        plotKM(cluster, filterClass(icaCensus), "ICA_CENSUS");
        plotKM(cluster, rpCar, "RP_CAR");
        plotKM(cluster, rpCensus, "RP_CENSUS");
    }

    public void plotEMWithFeature() throws Exception {
        FileFactory.TrainTest carTrainTest = fileFactory.getInstancesFromFile(ML.Files.CarBin, new Options(false, true));
        FileFactory.TrainTest censusTrainTest = fileFactory.getInstancesFromFile(ML.Files.CensusBin, new Options(false, true));
        FileFactory.TrainTest carBin = fileFactory.getInstancesFromFile(ML.Files.CarBin, new Options(false, true));
        FileFactory.TrainTest censusBin = fileFactory.getInstancesFromFile(ML.Files.CensusBin, new Options(false, true));

        Instances pcaCar = featureReductionService.applyPCAFilter(carTrainTest.train, 30);
        Instances pcaCensus = featureReductionService.applyPCAFilter(censusTrainTest.train, 30);
        Instances icaCar = featureReductionService.applyICA(carBin.test, 30);
        Instances icaCensus = featureReductionService.applyICA(censusBin.test, 30);
        Instances rpCar = featureReductionService.applyRP(carBin.train, 30);
        Instances rpCensus = featureReductionService.applyRP(censusBin.train, 30);

        Cluster em = new Cluster();
        em.setClusters(6);
        em.setIterations(1000);

        plotEM(em, pcaCar, "PCA_CAR");
        plotEM(em, pcaCensus, "PCA_CENSUS");
        plotEM(em, filterClass(icaCar), "ICA_CAR");
        plotEM(em, filterClass(icaCensus), "ICA_CENSUS");
        plotEM(em, rpCar, "RP_CAR");
        plotEM(em, rpCensus, "RP_CENSUS");
    }

    private Instances filterClass(Instances data) throws Exception{
        Remove filter = new Remove();
        filter.setAttributeIndices("" + (data.classIndex() + 1));
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }

    private MakeDensityBasedClusterer makeDensityBasedClustererWrapper(SimpleKMeans simpleKMeans, Instances data) throws Exception{
        MakeDensityBasedClusterer makeDensityBasedClusterer = new MakeDensityBasedClusterer();
        makeDensityBasedClusterer.setClusterer(simpleKMeans);
        makeDensityBasedClusterer.buildClusterer(data);
        return makeDensityBasedClusterer;
    }

    private Instances applyFeatureSelection(FileFactory.TrainTest data, Cluster cluster) throws Exception{
        switch (cluster.getFeatureSelection()){
            case ICA:
                return filterClass(featureReductionService.applyICA(data.test, 5));
            case PCA:
                return featureReductionService.applyPCAFilter(data.train, 5);
            case RP:
                return featureReductionService.applyRP(data.train, 5);
            case CFS:
                return null;
        }
        return null;
    }
}
