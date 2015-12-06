/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.udal.mains;

import cyber009.udal.functions.LinearFunction;
import cyber009.udal.functions.StatisticalAnalysis;
import cyber009.udal.libs.Variable;
import java.awt.BorderLayout;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

/**
 *
 * @author pavel
 */
public class WekaUDAL {
    
    public Variable data = null;
    public LinearFunction func = null;
    public Classifier classifier = null;
    
    public WekaUDAL() {
        
    }
    
    public void init(int nFeature, int nDataset) {
        func = new LinearFunction(System.currentTimeMillis());
        data = new Variable(nFeature, nDataset);
        func.generateSyntheticDataset(data);
        func.generateCoefficients(data.numberOfFeature);
    }
    
    public void activeLearning(int D) {
        Random r = new Random(System.currentTimeMillis());
        int index = 0;
        for(int d=0; d<D; d++) {
            index = r.nextInt(data.unLabelDataSets.numInstances());
            Instance set = data.unLabelDataSets.get(index);
            func.syntacticLabelFunction(set);
            data.labelDataSets.add(set);
            data.unLabelDataSets.remove(index);
        }
    }
    
    public void learnByClassifier() {
        try {
            data.labelDataSets.setClassIndex(data.numberOfFeature);
            classifier.buildClassifier(data.labelDataSets);
        } catch (Exception ex) {
            Logger.getLogger(WekaUDAL.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    
    public void forwardInstanceSelection() {
        double pp = 0.0D;
        Attribute classValus = data.labelDataSets.classAttribute();
        StatisticalAnalysis sa = new StatisticalAnalysis(classifier, data.labelDataSets);
        if(classValus.isNominal() == true) {
            for(Instance unLabelSet: data.unLabelDataSets) {
                pp = 0.0D;
                for (int i = 0; i < classValus.numValues(); i++) {
                    double classTarget = Double.parseDouble(classValus.value(i));
                    pp += sa.posteriorDistribution(unLabelSet, classTarget);
                    System.out.println(classTarget+ " "+ pp);
                }
                System.out.println(pp);
            }
        }
    }
    
    
    public void showPlot(Instances dataSet) {
        PlotData2D p2D = new PlotData2D(dataSet);
        p2D.setPlotName(dataSet.relationName());
        VisualizePanel vp = new VisualizePanel();
        vp.setName(dataSet.relationName());
        try {
            vp.addPlot(p2D);
            JFrame frame = new JFrame(dataSet.relationName());
            frame.setSize(600, 600);
            frame.setVisible(true);
            frame.getContentPane().setLayout(new BorderLayout());
            frame.getContentPane().add(vp, BorderLayout.CENTER);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setVisible(true);
        } catch (Exception ex) {
            Logger.getLogger(WekaUDAL.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void main(String[] args) {
        WekaUDAL udal = new WekaUDAL();
        // initial data
        udal.init(2, 17);
        udal.activeLearning(10);
        udal.classifier = new MultilayerPerceptron();
        ((MultilayerPerceptron)udal.classifier).setTrainingTime(100);
        udal.learnByClassifier();
        
        udal.forwardInstanceSelection();
        //System.out.println(udal.classifier.toString());
        //udal.showPlot(udal.data.labelDataSets);
        
        
    }
}
