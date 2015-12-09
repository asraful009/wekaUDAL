package cyber009.udal.libs;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import weka.core.Instances;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author pavel
 */
public class Variable {
    
    public int numberOfFeature;
    public int numberOfDataset;
    public Instances unLabelDataSets = null;
    public Instances labelDataSets = null;
    public HashMap<Integer, Double> infoFWunLabel = null;
    public List<Double> listOfClasses;

    /**
     *  <p>All variable's</p>
     * @param numberOfFeature
     * @param numberOfDataset 
     */
    public Variable(int numberOfFeature, int numberOfDataset) {
        this.numberOfFeature = numberOfFeature;
        this.numberOfDataset = numberOfDataset;
        this.listOfClasses = new ArrayList<>();
        this.infoFWunLabel = new HashMap<>();
    }
}
