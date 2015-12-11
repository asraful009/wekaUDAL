/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.udal.libs;

import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 *
 * @author asraful
 */
public class Utilitys {

    @SuppressWarnings("unchecked")
    public static Map sortByValue(Map unsortMap) {	 
	List list = new LinkedList(unsortMap.entrySet());
 
	Collections.sort(list, new Comparator() {
		public int compare(Object o1, Object o2) {
			return ((Comparable) ((Map.Entry) (o1)).getValue())
						.compareTo(((Map.Entry) (o2)).getValue());
		}
	});
 
	Map sortedMap = new LinkedHashMap();
	for (Iterator it = list.iterator(); it.hasNext();) {
		Map.Entry entry = (Map.Entry) it.next();
		sortedMap.put(entry.getKey(), entry.getValue());
	}
	return sortedMap;
}
    
}
