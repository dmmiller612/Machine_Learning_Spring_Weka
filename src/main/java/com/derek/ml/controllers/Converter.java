package com.derek.ml.controllers;

import com.derek.ml.models.ML;
import com.derek.ml.models.Options;
import com.derek.ml.services.FileFactory;
import com.derek.ml.services.LoadData;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

@Controller
public class Converter {

    @Autowired
    private FileFactory fileFactory;

    @Autowired
    private LoadData loadData;

    @ResponseBody
    @RequestMapping(value="/convert", method={RequestMethod.GET})
    public void doConvert() throws Exception{
        NominalToBinary nominalToBinary = new NominalToBinary();
        FileFactory.TrainTest data = fileFactory.handlePublicCensus(new Options(false, false));
        nominalToBinary.setInputFormat(data.test);
        Instances instances = Filter.useFilter(data.test, nominalToBinary);
        loadData.saveToArff(instances, "justATest2.arff");
    }
}
