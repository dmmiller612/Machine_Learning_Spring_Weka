package com.derek.ml.controllers;

import com.derek.ml.models.EMModel;
import com.derek.ml.models.Cluster;
import com.derek.ml.services.ClusterService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;


@Controller
public class Clustering {

    @Autowired
    private ClusterService clusterService;

    @ResponseBody
    @RequestMapping(value ="/kMeans", method={RequestMethod.GET})
    public String handleKmeans(Cluster cluster) throws Exception{
        return clusterService.handleKmeans(cluster);
    }

    @ResponseBody
    @RequestMapping(value = "/em", method = {RequestMethod.GET})
    public String handleEM(Cluster emModel) throws Exception{
        return clusterService.handleEM(emModel);
    }

    @ResponseBody
    @RequestMapping(value = "/em/plot", method = {RequestMethod.GET})
    public void handleEMPlot() throws Exception{
        clusterService.plotEMWithFeature();
    }

    @ResponseBody
    @RequestMapping(value = "/kMeans/plot", method = {RequestMethod.GET})
    public void handleKMPlot() throws Exception{
        clusterService.plotKMWithFeature();
    }
}
