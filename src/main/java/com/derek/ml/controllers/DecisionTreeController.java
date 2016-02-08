package com.derek.ml.controllers;

import com.derek.ml.models.DecisionTree;
import com.derek.ml.models.FileName;
import com.derek.ml.services.DecisionTreeService;
import com.derek.ml.services.FileFactory;
import com.derek.ml.services.LoadData;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import weka.core.Instances;


@Controller
public class DecisionTreeController {

    @Autowired
    DecisionTreeService decisionTreeService;

    @Autowired
    LoadData loadData;

    @Autowired
    FileFactory fileFactory;

    @ResponseBody
    @RequestMapping(value ="/decisiontree", method={RequestMethod.GET})
    public String getDecisionTreeAccuracy(DecisionTree decisionTree) throws Exception{
        return decisionTreeService.getDecisionTreeInformation(decisionTree);
    }

    @ResponseBody
    @RequestMapping(value="/decisiontree/test", method={RequestMethod.GET})
    public String testingError(DecisionTree decisionTree) throws Exception{
        return decisionTreeService.handleSplitData(decisionTree, 1, "");
    }

    @ResponseBody
    @RequestMapping(value="/decisiontree/model", method={RequestMethod.POST})
    public void createModel(DecisionTree decisionTree) throws Exception{
        decisionTreeService.createModel(decisionTree);
    }

    @ResponseBody
    @RequestMapping(value="/decisiontree/model", method={RequestMethod.GET})
    public String getModel(DecisionTree decisionTree) throws Exception{
        return decisionTreeService.getModel(decisionTree);
    }

    @ResponseBody
    @RequestMapping(value ="/createArff", method={RequestMethod.POST})
    public void createArff(@RequestBody FileName fileName) throws Exception{
        Instances instances = loadData.getDataFromCsvFile(fileName.getFileName() + ".csv");
        loadData.saveToArff(instances, fileName.getFileName() + ".arff");
    }


    @ResponseBody
    @RequestMapping(value ="/discretize", method={RequestMethod.POST})
    public void discretizeCensus() throws Exception{
        fileFactory.saveDiscretizedArff();
    }
}
