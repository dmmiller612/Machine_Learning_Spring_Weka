package com.derek.ml.controllers;

import com.derek.ml.models.NeuralNetworkModel;
import com.derek.ml.services.FileFactory;
import com.derek.ml.services.LoadData;
import com.derek.ml.services.NeuralNetworkService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class NeuralNetworkController {

    @Autowired
    private NeuralNetworkService neuralNetworkService;

    @ResponseBody
    @RequestMapping(value ="/neuralnetwork", method={RequestMethod.GET})
    public String neuralNetwork(NeuralNetworkModel neuralNetworkModel) throws Exception{
        return neuralNetworkService.handleNeuralNetwork(neuralNetworkModel);
    }

    @ResponseBody
    @RequestMapping(value = "/neuralnetwork/test", method={RequestMethod.GET})
    public String neuralnetworkTest(NeuralNetworkModel neuralNetworkModel) throws Exception{
        return neuralNetworkService.handleSplitData(neuralNetworkModel, 1, "");
    }

    @ResponseBody
    @RequestMapping(value="/neuralnetwork/model", method={RequestMethod.POST})
    public void createModel(NeuralNetworkModel nn) throws Exception{
        neuralNetworkService.createModel(nn);
    }

    @ResponseBody
    @RequestMapping(value="/neuralnetwork/model", method={RequestMethod.GET})
    public String getModel(NeuralNetworkModel nn) throws Exception{
        return neuralNetworkService.getModel(nn);
    }

}
