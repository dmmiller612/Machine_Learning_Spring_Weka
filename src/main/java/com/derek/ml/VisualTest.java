package com.derek.ml;

import java.awt.BorderLayout;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.BufferedReader;
import java.io.FileReader;

import javax.swing.JFrame;

import com.derek.ml.models.DecisionTree;
import com.derek.ml.services.DecisionTreeService;
import com.derek.ml.services.FileFactory;
import com.derek.ml.services.LoadData;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class VisualTest {

    //private static DecisionTreeService decisionTreeService = new DecisionTreeService();
/*
    public static void main(String args[]) throws Exception {
        decisionTreeService.fileFactory = new FileFactory();
        decisionTreeService.fileFactory.loadData = new LoadData();
        DecisionTree decisionTree = new DecisionTree();
        decisionTree.setMinNumObj(10);
        J48 cls = decisionTreeService.handlePublicJ48(decisionTree);

        // display classifier
        final javax.swing.JFrame jf =
                new javax.swing.JFrame("Weka Classifier Tree Visualizer: J48");
        jf.setSize(4000,1000);
        jf.getContentPane().setLayout(new BorderLayout());
        TreeVisualizer tv = new TreeVisualizer(null,
                cls.graph(),
                new PlaceNode2());
        jf.getContentPane().add(tv, BorderLayout.CENTER);
        jf.addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
            }
        });

        jf.setVisible(true);
        tv.fitToScreen();
    }
    */

}
