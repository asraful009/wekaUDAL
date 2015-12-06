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
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author pavel
 */
public class StatisticalAnalysis {
    
    Evaluation evaluation = null;
    Classifier classifier = null;
    
    public StatisticalAnalysis(Classifier classifier, Instances tranDataSet) {
        try {
            this.classifier = classifier;
            evaluation = new Evaluation(tranDataSet);
        } catch (Exception ex) {
            Logger.getLogger(StatisticalAnalysis.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    
    public double posteriorDistribution(Instance unLabelSet, double classTarget) {
        double prDistribution = 0.0D;
        try {
            double classPradic = evaluation.evaluateModelOnceAndRecordPrediction(classifier, unLabelSet);
            prDistribution = Math.abs(classPradic-classTarget);
        } catch (Exception ex) {
            Logger.getLogger(StatisticalAnalysis.class.getName()).log(Level.SEVERE, null, ex);
        }
        return prDistribution;
    }
}
