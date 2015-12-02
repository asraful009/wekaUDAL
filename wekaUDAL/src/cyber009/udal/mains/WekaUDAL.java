/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.udal.mains;

import cyber009.udal.functions.LinearFunction;
import cyber009.udal.libs.Variable;
import weka.core.Instance;

/**
 *
 * @author pavel
 */
public class WekaUDAL {
    
    public Variable data = null;
    public LinearFunction func = null;
    
    public WekaUDAL() {
        
    }
    
    public void init(int nFeature, int nDataset) {
        func = new LinearFunction(System.currentTimeMillis());
        data = new Variable(nFeature, nDataset);
        func.generateSyntheticDataset(data);
        func.generateCoefficients(data.numberOfFeature);
    }
    
    public void activeLearning(int D) {
        for(int d=0; d<D; d++) {
            Instance set = data.unLabelDataSets.get(d);
            func.syntacticLabelFunction(set);
            data.labelDataSets.add(set);
            data.unLabelDataSets.remove(d);
        }
    }
    
    
    
    public static void main(String[] args) {
        WekaUDAL udal = new WekaUDAL();
        udal.init(5, 10);
        System.out.println(udal.data.unLabelDataSets);
        udal.activeLearning(3);
        System.out.println(udal.data.unLabelDataSets);
        System.out.println(udal.data.labelDataSets);
    }
}
