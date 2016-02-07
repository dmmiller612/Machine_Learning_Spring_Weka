package com.derek.ml.controllers;

import com.derek.ml.models.NearestNeighbor;
import com.derek.ml.services.FileFactory;
import com.derek.ml.services.KNNService;
import com.derek.ml.services.LoadData;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class KNNController {

    @Autowired
    private KNNService knnService;

    @ResponseBody
    @RequestMapping(value ="/knn", method={RequestMethod.GET})
    public String knnClassify(NearestNeighbor nearestNeighbor) throws Exception{
        return knnService.handleKNNService(nearestNeighbor);
    }

    @ResponseBody
    @RequestMapping(value ="/knn/test", method={RequestMethod.GET})
    public String knnTesting(NearestNeighbor nearestNeighbor) throws Exception{
        return knnService.handleSplitData(nearestNeighbor, 1, "");
    }

    @ResponseBody
    @RequestMapping(value="/knn/model", method={RequestMethod.POST})
    public void createModel(NearestNeighbor nearestNeighbor) throws Exception{
        knnService.createModel(nearestNeighbor);
    }

    @ResponseBody
    @RequestMapping(value="/knn/model", method={RequestMethod.GET})
    public String getModel(NearestNeighbor nearestNeighbor) throws Exception{
        return knnService.getModel(nearestNeighbor);
    }
}
