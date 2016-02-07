package com.derek.ml;

import com.derek.ml.models.*;
import com.derek.ml.services.DecisionTreeService;
import com.derek.ml.services.KNNService;
import com.derek.ml.services.NeuralNetworkService;
import com.derek.ml.services.SVMService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.web.WebMvcAutoConfiguration;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;

@SpringBootApplication
public class Server extends WebMvcAutoConfiguration {

    public static void main (String[] args) throws Exception {
        ApplicationContext ctx = new SpringApplication(Server.class).run(args);
        System.out.println("MACHINE LEARNING ASSIGNMENT 1 IS RUNNING");

        DecisionTreeService decisionTreeService = ctx.getBean(DecisionTreeService.class);
        KNNService knnService = ctx.getBean(KNNService.class);
        SVMService svmService = ctx.getBean(SVMService.class);
        NeuralNetworkService neuralNetworkService = ctx.getBean(NeuralNetworkService.class);

        /**
         * Car Evaluation Dataset
         */
        System.out.println("\n\n ============================ \n Car Evaluation Dataset \n ========================");

        //car evaluation decision tree
        System.out.println("\n\n ===================== \n Car Evaluation Decision Tree model. MinNumObj=2 AND Confidence =.25 \n===========================\n");
        System.out.println(decisionTreeService.getModel(new DecisionTree(2, "0.25", false, ML.Files.Car)));

        //car evaluation boosted decision tree
        System.out.println("\n\n ===================== \n Car Evaluation Boosted Decision Tree Model. MinNumObj=2 AND Confidence=.25 \n" +
                "===========================\n");
        System.out.println(decisionTreeService.getModel(new DecisionTree(2, "0.25", true, ML.Files.Car)));

        //car evaluation knn
        System.out.println("\n\n ===================== \n Car KNN model. k=3 \n ===========================\n");
        System.out.println(knnService.getModel(new NearestNeighbor(3, ML.Files.Car)));

        //car evaluation knn
        System.out.println("\n\n ===================== \n Car KNN model. k=20 \n ===========================\n");
        System.out.println(knnService.getModel(new NearestNeighbor(20, ML.Files.Car)));

        //car evaluation Neural Network
        System.out.println("\n\n ===================== \n Car Neural Network model. Epoch Rate=500 and Hidden Layers=10 \n ===========================\n");
        System.out.println(neuralNetworkService.getModel(new NeuralNetworkModel(10, 500, ML.Files.Car)));

        //car evaluation Neural Network
        System.out.println("\n\n ===================== \n Car Neural Network model. Epoch Rate=500 and Hidden Layers=3 \n ===========================\n");
        System.out.println(neuralNetworkService.getModel(new NeuralNetworkModel(3, 500, ML.Files.Car)));

        //car evaluation SVM
        System.out.println("\n\n ===================== \n Car SVM model. Kernel = Linear \n ===========================\n");
        System.out.println(svmService.getModel(new SVMModel(SVMModel.KernelType.Linear, ML.Files.Car)));

        //car evaluation SVM
        System.out.println("\n\n ===================== \n Car SVM model. Kernel = Polynomial \n ===========================\n");
        System.out.println(svmService.getModel(new SVMModel(SVMModel.KernelType.Polynomial, ML.Files.Car)));

        //car evaluation SVM
        System.out.println("\n\n ===================== \n Car SVM model. Kernel = Sigmoid \n ===========================\n");
        System.out.println(svmService.getModel(new SVMModel(SVMModel.KernelType.Sigmoid, ML.Files.Car)));

        /**
         * CENSUS DATASET
         */
        System.out.println("\n\n =================================================================== \n");
        System.out.println("\n\n====================== \n Census Dataset \n ========================");
        System.out.println("\n =================================================================== \n\n");

        //census decision tree
        System.out.println("\n\n ===================== \n Census Decision Tree model. MinNumObj=100 AND Confidence =.25 \n===========================\n");
        System.out.println(decisionTreeService.getModel(new DecisionTree(100, "0.25", false, ML.Files.Census)));

        //census boosted decision tree
        System.out.println("\n\n ===================== \n Census Boosted Decision Tree model. MinNumObj=100 AND Confidence =.25 \n===========================\n");
        System.out.println(decisionTreeService.getModel(new DecisionTree(100, "0.25", true, ML.Files.Census)));

        //census evaluation knn
        System.out.println("\n\n ===================== \n Census KNN model. k=3 \n ===========================\n");
        System.out.println("\n Loading.... \n");
        System.out.println(knnService.getModel(new NearestNeighbor(3, ML.Files.Census)));

        //census evaluation knn
        System.out.println("\n\n ===================== \n Census KNN model. k=5 \n ===========================\n");
        System.out.println("\n Loading.... \n");
        System.out.println(knnService.getModel(new NearestNeighbor(5, ML.Files.Census)));

        //census neural network
        System.out.println("\n\n ===================== \n Census Neural Network model. Epoch Rate=250 and Hidden Layers=10 \n ===========================\n");
        System.out.println("\n Loading.... \n");
        System.out.println(neuralNetworkService.getModel(new NeuralNetworkModel(10, 250, ML.Files.Census)));

        //census Neural Network
        System.out.println("\n\n ===================== \n Census Neural Network model. Epoch Rate=500 and Hidden Layers=3 \n ===========================\n");
        System.out.println("\n Loading.... \n");
        System.out.println(neuralNetworkService.getModel(new NeuralNetworkModel(3, 500, ML.Files.Census)));

        //Census SVM
        System.out.println("\n\n ===================== \n Census SVM model. Kernel = Linear \n ===========================\n");
        System.out.println("\n Loading.... \n");
        System.out.println(svmService.getModel(new SVMModel(SVMModel.KernelType.Linear, ML.Files.Census)));

        //Census SVM
        System.out.println("\n\n ===================== \n Census SVM model. Kernel = Sigmoid \n ===========================\n");
        System.out.println("\n Loading.... \n");
        System.out.println(svmService.getModel(new SVMModel(SVMModel.KernelType.Sigmoid, ML.Files.Census)));

        //Census SVM
        System.out.println("\n\n ===================== \n Census SVM model. Kernel = RBF \n ===========================\n");
        System.out.println("\n Loading.... \n");
        System.out.println(svmService.getModel(new SVMModel(SVMModel.KernelType.RBF, ML.Files.Census)));

        System.out.println("\n\n =================================================================== \n");
        System.out.println("\n\n====================== \n Census Dataset With Feature Selection \n ========================");
        System.out.println("\n =================================================================== \n\n");

        //census Neural Network Feature Selection
        System.out.println("\n\n ===================== \n Census Neural Network model with Feature selection. Epoch Rate=250 and Hidden Layers=10 \n ===========================\n");
        System.out.println("\n Loading.... \n");
        System.out.println(neuralNetworkService.getModel(new NeuralNetworkModel(10, 250, ML.Files.Census, true)));

        //census Neural Network Feature Selection
        System.out.println("\n\n ===================== \n Census Neural Network model with Feature selection. Epoch Rate=250 and Hidden Layers=5 \n ===========================\n");
        System.out.println("\n Loading.... \n");
        System.out.println(neuralNetworkService.getModel(new NeuralNetworkModel(5, 250, ML.Files.Census, true)));

        //census evaluation knn with feature selection
        System.out.println("\n\n ===================== \n Census KNN model with feature selection. k=5 \n ===========================\n");
        System.out.println("\n Loading.... \n");
        System.out.println(knnService.getModel(new NearestNeighbor(5, ML.Files.Census, true)));

        //census evaluation knn with feature selection
        System.out.println("\n\n ===================== \n Census KNN model with feature selection. k=3 \n ===========================\n");
        System.out.println("\n Loading.... \n");
        System.out.println(knnService.getModel(new NearestNeighbor(3, ML.Files.Census, true)));

        System.out.println("\n \n ================================ \n FINISHED! \n ==============================");

    }
}

