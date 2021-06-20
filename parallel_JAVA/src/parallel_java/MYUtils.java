/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package parallel_java;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author malega
 */
public class MYUtils {
    public static void SaveResultsToCSV(int[] threads_count, List<String> resolutions, List<float[]> durations_total, MorphOp operation ){
        
        StringBuilder sb = new StringBuilder();
        sb.append("Resolution");
        sb.append(",");
        //Append Thread numbers as columns
        for (int i = 0; i < threads_count.length; i++) {
            sb.append(threads_count[i]);
            if (i != threads_count.length - 1) {
                sb.append(",");
            }
        }
        sb.append("\n");
        for (int i = 0; i < resolutions.size(); i++) {
            sb.append(resolutions.get(i));
            sb.append(",");
            float[] values = durations_total.get(i);
            for (int k = 0; k < values.length; k++) {
                sb.append(values[k]);
                if (k != values.length - 1) {
                    sb.append(",");
                }
            }
            sb.append("\n");
        }
        
        MYUtils.Write("../results_JAVA/timings_java_"+(operation==MorphOp.Dilation ? "dilation" : "erosion")+"_optimized.csv", sb);
    }
    
    public static void Write(String filename, StringBuilder sb) {
        try (PrintWriter writer = new PrintWriter(new File(filename))) {
            writer.write(sb.toString());
            System.out.println("CSV written!");
        } catch (FileNotFoundException e) {
            System.out.println(e.getMessage());
        }
    }
    
    public static File[] GetResolutionFiles(File images_folder){
        File[] listOfResolutions = images_folder.listFiles();
        Arrays.sort(listOfResolutions);
        return listOfResolutions;
    }
}
