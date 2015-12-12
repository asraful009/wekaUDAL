/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.udal.mains;

import cyber009.udal.functions.LinearFunction;
import cyber009.udal.functions.StatisticalAnalysis;
import cyber009.udal.libs.Utilitys;
import cyber009.udal.libs.Variable;
import java.awt.BorderLayout;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.AttributeStats;
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
            try {
                index = r.nextInt(data.unLabelDataSets.numInstances());
            } catch(Exception e) {
                //System.out.println("#"+data.unLabelDataSets.numInstances());
                break;
            }
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
    
//    public void addToMinQueue(int index) {
//        int j=0;
//        int temp = -1;
//        for(j= MINQUEUESIZE-1; j >=0; j--) {
//            if(MinQueue[j] != -1) {
//                if(v.X_FL[MinQueue[j]] > v.X_FL[index]) {
//                    if(j != MINQUEUESIZE -1) {
//                        MinQueue[j+1] = MinQueue[j];
//                    }
//                } else {
//                    break;
//                }
//            }
//            temp = j;
//        }
//        if(temp != -1) {
//            MinQueue[temp] = index;
//        }
//    }
    
    public void forwardInstanceSelection() {
        double pp = 0.0D;
        AttributeStats classStats = data.labelDataSets.attributeStats(data.labelDataSets.classIndex());
        StatisticalAnalysis sa = new StatisticalAnalysis();
        if(classStats.nominalCounts != null) {
            for(int n=0; n<data.unLabelDataSets.numInstances(); n++) {
                Instance unLabelSet =  data.unLabelDataSets.get(n);
                pp = 0.0D;
                for (int i = 0; i < classStats.nominalCounts.length; i++) {
                    double classTarget = new Double(
                            data.labelDataSets.attribute(
                                data.labelDataSets.classIndex()).value(i));
                    unLabelSet.setClassValue(classTarget);
                    pp += sa.posteriorDistribution(classifier, data.labelDataSets, 
                            unLabelSet, classTarget);
                    pp *= sa.conditionalEntropy(classifier, data.labelDataSets, 
                            data.unLabelDataSets, unLabelSet, classTarget);
                    unLabelSet.setClassValue(Double.NaN);
                }
                data.infoFWunLabel.put(n, pp);
            }            
        }
    }
    
    @SuppressWarnings("unchecked")
    public void updateLabelDataSet() {
        int count = 0;
        Instances temp = new Instances(data.unLabelDataSets, 0, 0);
        data.infoFWunLabel = 
                (HashMap<Integer, Double>) Utilitys.sortByValue((Map)data.infoFWunLabel);
        for (Map.Entry<Integer, Double> entrySet : data.infoFWunLabel.entrySet()) {
            int index = entrySet.getKey();
            if(count < data.N_FL) {
//                System.out.println(index + " : "
//                        +entrySet.getValue() + " : "
//                        + data.unLabelDataSets.instance(index).toString());
                func.syntacticLabelFunction(data.unLabelDataSets.get(index));
                data.labelDataSets.add(data.unLabelDataSets.get(index));
            } else {
                temp.add(data.unLabelDataSets.get(index));
            }
            count++;
        }
        data.infoFWunLabel.clear();
        data.unLabelDataSets.clear();
        data.unLabelDataSets.addAll(temp);
        //System.out.println("------------------------------------------");
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
        udal.init(2, 3000);
        udal.data.N_FL = 30;
        udal.activeLearning(70);
        udal.showPlot(udal.data.labelDataSets);
        // forward Instance Selection
        for(int i=0; i<20; i++) {
            System.out.println("---------itaration :"+i+" -------------");
            udal.classifier = new MultilayerPerceptron();
            ((MultilayerPerceptron)udal.classifier).setTrainingTime(10000);
            System.out.println(udal.data.labelDataSets.numInstances()
            //        +"\n"+udal.data.labelDataSets
            );
            System.out.println(udal.data.unLabelDataSets.numInstances()
            //        +"\n"+udal.data.unLabelDataSets
            );
            udal.learnByClassifier();
            udal.forwardInstanceSelection();
            udal.updateLabelDataSet();
            
        }        
        //System.out.println(udal.classifier.toString());
        udal.showPlot(udal.data.labelDataSets);
        
        
    }
}
