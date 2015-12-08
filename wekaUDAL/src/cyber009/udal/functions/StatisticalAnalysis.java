/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.udal.functions;

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author pavel
 */
public class StatisticalAnalysis {
    
   
    
    public StatisticalAnalysis() {
        
    }
    
    public double probabilityOfTargerClass(Instances dataSet, double classTarget) {
        AttributeStats classStats = dataSet.attributeStats(dataSet.classIndex());
        double ptc = 0.0D;
        if(classStats.nominalCounts != null ) {
            for(int i=0; i<classStats.nominalCounts.length; i++) {
                if(new Double(
                        dataSet.attribute(
                                dataSet.classIndex()).value(i)) == classTarget) {
                     ptc = (double)classStats.nominalCounts[i]/(double)classStats.totalCount;
                }
            }
        }
        return ptc;
    }
    
    public double posteriorDistribution(Classifier classifier, Instances trainingDataSet,
            Instance unLabelSet, double classTarget) {
        double prDistribution = 0.0D;
        Evaluation evaluation = null;
        AttributeStats classStats = trainingDataSet.attributeStats(trainingDataSet.classIndex());
        try {
            evaluation = new Evaluation(trainingDataSet);
            evaluation.evaluateModelOnceAndRecordPrediction(classifier, unLabelSet);
            double classPradic = evaluation.pctCorrect(); // must be show for correctness  ----------------------
            prDistribution = classPradic
                    *probabilityOfTargerClass(trainingDataSet, classTarget);
            //System.out.println(evaluation.pctCorrect());
        } catch (Exception ex) {
            Logger.getLogger(StatisticalAnalysis.class.getName()).log(Level.SEVERE, null, ex);
        }
        return prDistribution;
    }
    
    public double conditionalEntropy() {
        double cEnt = 0.0D;
        return cEnt;
    }
}
