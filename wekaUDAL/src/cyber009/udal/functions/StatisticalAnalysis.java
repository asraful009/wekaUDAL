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
    
    public static boolean instanceCMPWithoutClass(Instance a, Instance b) {
        if(a.numAttributes() == b.numAttributes()) {
            for(int i = 0; i<a.numAttributes(); i++) {
                if(a.value(i) != b.value(i)) {
                    return false;
                }
            }
        } else {
            return false;
        }
        return true;
    }
    
    /**
     * 
     * @param dataSet
     * @param classTarget
     * @return 
     */
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
    
    /**
     * 
     * @param classifier
     * @param trainingDataSet
     * @param unLabelSet
     * @param classTarget
     * @return 
     */
    public double posteriorDistribution(Classifier classifier, Instances trainingDataSet,
            Instance unLabelSet, double classTarget) {
        double prDistribution = 0.0D;
        double classPradic = 0.0D;
        Evaluation evaluation = null;
        try {
            evaluation = new Evaluation(trainingDataSet);
            evaluation.evaluateModelOnceAndRecordPrediction(classifier, unLabelSet);
            //evaluation.pctCorrect(); // must be show for correctness  ----------------------
            classPradic = evaluation.meanAbsoluteError();
            prDistribution = classPradic
                    *probabilityOfTargerClass(trainingDataSet, classTarget);
//            System.out.println(classPradic);
        } catch (Exception ex) {
            Logger.getLogger(StatisticalAnalysis.class.getName()).log(Level.SEVERE, null, ex);
        }
        return prDistribution;
    }
    
    /**
     * 
     * @param classifier
     * @param trainingDataSet
     * @param unLabelDataSets
     * @param unLabelSet
     * @param classTarget
     * @return 
     */
    public double conditionalEntropy(Classifier classifier, Instances trainingDataSet,
            Instances unLabelDataSets, Instance unLabelSet, double classTarget) {
        double cEnt = 0.0D;
        double entropy = 0.0D;
        unLabelSet.setClassValue(classTarget);
        trainingDataSet.add(trainingDataSet.numInstances(),unLabelSet);
        AttributeStats classStats = trainingDataSet.attributeStats(trainingDataSet.classIndex());
        for(Instance set: unLabelDataSets) {
            if(instanceCMPWithoutClass(set, unLabelSet) == true)
                continue;
            for (int i = 0; i < classStats.nominalCounts.length; i++) {
                double target = new Double(
                        trainingDataSet.attribute(trainingDataSet.classIndex()).value(i));
                set.setClassValue(target);
                entropy = posteriorDistribution(classifier,
                                trainingDataSet, set, classTarget);
                //System.out.println("entropy:"+entropy);
                cEnt += -(entropy)*Math.log10(entropy);
                set.setClassMissing();
            }
        }
        trainingDataSet.remove(trainingDataSet.numInstances()-1);
        return cEnt;
    }
}
