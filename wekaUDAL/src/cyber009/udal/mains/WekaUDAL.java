/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.udal.mains;

import cyber009.udal.functions.LinearFunction;
import cyber009.udal.libs.Variable;

/**
 *
 * @author pavel
 */
public class WekaUDAL {
    
    public Variable data = null;
    public LinearFunction func = null;
    
    public WekaUDAL() {
        
    }
    
    public void init() {
        func = new LinearFunction(System.currentTimeMillis());
        data = new Variable(5, 100);
        func.generateSyntheticDataset(data);
    }
    
    
    public static void main(String[] args) {
        WekaUDAL udal = new WekaUDAL();
        udal.init();
        System.out.println(udal.data.allDataSets);
    }
}
