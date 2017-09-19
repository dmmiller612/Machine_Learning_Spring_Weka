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
        System.out.println("MACHINE LEARNING IS RUNNING");
    }
}

