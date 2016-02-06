package com.derek.ml.controllers;

import com.derek.ml.models.SVMModel;
import com.derek.ml.services.SVMService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class SVMController {

    @Autowired
    private SVMService svmService;

    @ResponseBody
    @RequestMapping(value ="/svm", method={RequestMethod.GET})
    public String svmHandle(SVMModel svmModel) throws Exception{
        return svmService.handleSVM(svmModel);
    }

    @ResponseBody
    @RequestMapping(value = "/svm/test", method = {RequestMethod.GET})
    public String svmTestHandle(SVMModel svmModel) throws Exception{
        return svmService.handleSplitData(svmModel, 1, "");
    }

    @ResponseBody
    @RequestMapping(value="/svm/model", method={RequestMethod.POST})
    public void createModel(SVMModel svmModel) throws Exception{
        svmService.createModel(svmModel);
    }

    @ResponseBody
    @RequestMapping(value="/svm/model", method={RequestMethod.GET})
    public String getModel(SVMModel svmModel) throws Exception{
        return svmService.getModel(svmModel);
    }

}
