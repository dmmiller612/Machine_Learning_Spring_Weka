package com.derek.ml.controllers;

import com.derek.ml.services.FeatureReductionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class FeatureReductionController {

    @Autowired
    private FeatureReductionService featureReductionService;

    @ResponseBody
    @RequestMapping(value ="/featureReduction/pca", method={RequestMethod.GET})
    public String handleFeatureReduction() throws Exception{
        return featureReductionService.handlePCAFeatures();
    }

    @ResponseBody
    @RequestMapping(value ="/featureReduction/rp", method={RequestMethod.GET})
    public String handleRP() throws Exception{
        return featureReductionService.handleRandomizedProjectionFeatures();
    }

    @ResponseBody
    @RequestMapping(value ="/featureReduction/ica", method={RequestMethod.GET})
    public String handleICA() throws Exception{
        return featureReductionService.handleICAFeatures();
    }

    @ResponseBody
    @RequestMapping(value ="/featureReduction/cfs", method={RequestMethod.GET})
    public String handleCfsSubsetEval() throws Exception{
        return featureReductionService.handleCFSSubsetEval();
    }

    @ResponseBody
    @RequestMapping(value ="/featureReduction/rp/plot", method={RequestMethod.GET})
    public void handlePlotRp() throws Exception{
        featureReductionService.plotRP();
    }

    @ResponseBody
    @RequestMapping(value ="/featureReduction/pca/plot", method={RequestMethod.GET})
    public void handlePlotPCA() throws Exception{
        featureReductionService.plotPCA();
    }

    @ResponseBody
    @RequestMapping(value ="/featureReduction/ica/plot", method={RequestMethod.GET})
    public void handlePlotICA() throws Exception{
        featureReductionService.plotICA();
    }
}
