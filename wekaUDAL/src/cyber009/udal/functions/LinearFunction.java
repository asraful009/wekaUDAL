/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.udal.functions;

import cyber009.udal.libs.Variable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author pavel
 */
public class LinearFunction {
    Random rand = null;
    public double [] coefficients =null;
    
    public LinearFunction(long randSeed) {
        rand = new Random(randSeed);        
    }
    /**
     * <p>use for generate Synthetic Dataset</p>
     * @param data 
     */
    public void generateSyntheticDataset(Variable data) {
        List<Attribute> atts = new ArrayList<>();
        for(int n=0; n<data.numberOfFeature; n++) {
            atts.add(new Attribute("X"+n));
        }
        List<String> classValus = new ArrayList<>();
        classValus.add("1");
        classValus.add("0");
        atts.add(new Attribute("class", classValus));
        data.unLabelDataSets = new Instances("Syn Data unlabel data set:"+data.numberOfDataset, (ArrayList<Attribute>) atts, data.numberOfDataset);
        data.labelDataSets = new Instances("Syn Data label data set:"+data.numberOfDataset, (ArrayList<Attribute>) atts, data.numberOfDataset);
        Instance set = null;
        for(int d = 0; d<data.numberOfDataset; d++) {
            set = new DenseInstance(data.numberOfFeature+1);
            for(int n = 0; n<data.numberOfFeature; n++) {
                set.setValue(n, rand.nextGaussian());
            }
            //set.setValue(data.numberOfFeature, ); // class value empty does not set any thing that put ? unlabel data set
            data.unLabelDataSets.add(set);
        }
        data.unLabelDataSets.setClassIndex(data.numberOfFeature);
    }
    
    public void generateCoefficients(int numberOfFeature) {
        coefficients = new double[numberOfFeature];
        for(int n=0; n<numberOfFeature; n++) {
            coefficients[n] = rand.nextGaussian();
        }
    }
    
    public void syntacticLabelFunction(Instance set) {
        double sum = 0.0D;
        for(int n=0; n<set.numAttributes()-1; n++) {
            sum+= set.value(n)*coefficients[n];
        }
        if(sum<0.0D) {
            set.setClassValue("1");
        } else {
            set.setClassValue("0");
        }
    }
}
