/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package parallel_java;
//import java.lang.*;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;


/**
 *
 * @author Fabian Greavu
 */
public class Morphology {
    /**
     * Executes erosion on a binary image.
     * @param img
     * @param ExecuteOnBackgroundPixel
     * @return 
     */
    public static MyOwnImage Erosion_binary(MyOwnImage img, boolean ExecuteOnBackgroundPixel){
        return Erosion_binary(img, ExecuteOnBackgroundPixel, Runtime.getRuntime().availableProcessors());
    }
    
    /**
     * Executes erosion on a binary image.
     * @param img
     * @param ExecuteOnBackgroundPixel
     * @param num_ths
     * @return 
     */
    public static MyOwnImage Erosion_binary(MyOwnImage img, boolean ExecuteOnBackgroundPixel, int num_ths){
        return Erosion_binary(img, ExecuteOnBackgroundPixel, num_ths, new int[]{1,1,1,1,1,1,1,1,1}, 3);
    }
    
    /**
     * Executes erosion on a binary image.
     * @param img
     * @param ExecuteOnBackgroundPixel
     * @param num_ths
     * @param mask
     * @param maskSize
     * @return 
     */
    public static MyOwnImage Erosion_binary(MyOwnImage img, boolean ExecuteOnBackgroundPixel, int num_ths, int mask[], int maskSize){
        return ExecuteMaskOnBinaryImage(img, num_ths, mask, maskSize, MorphOp.Erosion, ExecuteOnBackgroundPixel);
    }
    
    /**
     * Executes dilation on a binary image.
     * @param img
     * @param ExecuteOnBackgroundPixel
     * @return 
     */
    public static MyOwnImage Dilation_binary(MyOwnImage img, boolean ExecuteOnBackgroundPixel){
        return Dilation_binary(img, ExecuteOnBackgroundPixel, Runtime.getRuntime().availableProcessors());
    }
    
    /**
     * Executes dilation on a binary image.
     * @param img
     * @param ExecuteOnBackgroundPixel
     * @param num_ths
     * @return 
     */
    public static MyOwnImage Dilation_binary(MyOwnImage img, boolean ExecuteOnBackgroundPixel, int num_ths){
        return Dilation_binary(img, ExecuteOnBackgroundPixel, num_ths, new int[]{1,1,1,1,1,1,1,1,1}, 3);
    }
    
    /**
     * Executes dilation on a binary image.
     * @param img
     * @param ExecuteOnBackgroundPixel
     * @param num_ths
     * @param mask
     * @param maskSize
     * @return 
     */
    public static MyOwnImage Dilation_binary(MyOwnImage img, boolean ExecuteOnBackgroundPixel, int num_ths, int mask[], int maskSize){
        return ExecuteMaskOnBinaryImage(img, num_ths, mask, maskSize, MorphOp.Dilation, ExecuteOnBackgroundPixel);
    }
    
    
    private static MyOwnImage ExecuteMaskOnBinaryImage(MyOwnImage img, int num_ths, int mask[], int maskSize, MorphOp operation, boolean ExecuteOnBackgroundPixel){
        int width = img.getImageWidth();
        int height = img.getImageHeight();
        num_ths = checkThreadsNum(height, num_ths);
        
        //Save output of dilation
        int output[] = new int[width * height];
        ExecutorService taskExecutor = Executors.newFixedThreadPool(num_ths);
        int rows_per_thread = (int)Math.floor((double)(height/num_ths));
        
        //Populate pool
        for(int y = 0; y < height; y += rows_per_thread){
            BinaryMaskApplier t = new BinaryMaskApplier(img, output, y, Math.min(y + rows_per_thread, height), width, height, mask, maskSize, operation, ExecuteOnBackgroundPixel);
            taskExecutor.execute(t);
        }
        
        //Barrier
        waitExecutors(taskExecutor);
        
        //Copy results to new image
        return CreateImage(new MyOwnImage(img), output);
    }
    
    private static MyOwnImage CreateImage(MyOwnImage image, int[] output){
        int width = image.getImageWidth();
        int height = image.getImageHeight();
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                int v = output[x+y*width];
                image.setPixel(x, y, 255, v, v, v);
            }
        }
        return image;
    }
    
    /**
     * Wait for an ExecutorService to finish all executors.
     * @param taskExecutor The started service
     */
    private static void waitExecutors(ExecutorService taskExecutor){
        taskExecutor.shutdown();
        try {
          taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
          System.err.println("join() interrupted: " + e.toString());
        }
    }
    
    /**
     * Returns integer clamped value between 0 and max
     * @param num target number
     * @param max max value
     * @return 
     */
    private static int checkThreadsNum(int num, int max){
        return Math.max(0, Math.min(max, num));
    }
    
        
    /**
     * This method will perform dilation operation on the grayscale image img. with default mask and on CPU cores.
     * 
     * @param img The image on which dilation operation is performed
     * @return copy of dilated image
     */
    public static MyOwnImage Dilation_grayscale(MyOwnImage img){
        return Dilation_grayscale(img, Runtime.getRuntime().availableProcessors()); //Assume majority of hardware have 4 cores at least
    }
    
    /**
     * This method will perform dilation operation on the grayscale image img with given thread number.
     * 
     * @param img The image on which dilation operation is performed
     * @param num_ths Number of threads to start with
     * @return copy of dilated image
     */
    public static MyOwnImage Dilation_grayscale(MyOwnImage img, int num_ths){
        return Dilation_grayscale(img, num_ths, new int[]{1,1,1,1,1,1,1,1,1}, 3);
    }
    
    /**
     * This method will perform dilation operation on the grayscale image img.It will find the maximum value among the pixels that are under the mask [element value 1] and will
 set the origin to the maximum value.
     * 
     * @param img The image on which dilation operation is performed
     * @param num_ths Number of threads to run on
     * @param mask the square mask.
     * @param maskSize the size of the square mask. [i.e., number of rows]
     * @return copy of dilated image
     */
    public static MyOwnImage Dilation_grayscale(MyOwnImage img, int num_ths, int mask[], int maskSize){
        return ExecuteMaskOnImage(img, num_ths, mask, maskSize, MorphOp.Dilation);
    }
    
    private static MyOwnImage ExecuteMaskOnImage(MyOwnImage img, int num_ths, int mask[], int maskSize, MorphOp operation){
        int width = img.getImageWidth();
        int height = img.getImageHeight();
        num_ths = checkThreadsNum(height, num_ths);
        
        //Save output of dilation
        int output[] = new int[width*height];
        ExecutorService taskExecutor = Executors.newFixedThreadPool(num_ths);
        int rows_per_thread = (int)Math.floor((double)(height/num_ths));
        
        //Populate pool
        for(int y = 0; y < height; y += rows_per_thread){
            MaskApplier t = new MaskApplier(img, output, y, Math.min(y + rows_per_thread, height), width, height, mask, maskSize, operation);
            taskExecutor.execute(t);
        }
        
        //Barrier
        waitExecutors(taskExecutor);
        
        //Copy results to new image
        return CreateImage(new MyOwnImage(img), output);
    }
    
    /**
     * This method will perform erosion operation on the grayscale image img. with default mask and on CPU cores.
     * 
     * @param img The image on which dilation operation is performed
     * @return copy of dilated image
     */
    public static MyOwnImage Erosion_grayscale(MyOwnImage img){
        return Erosion_grayscale(img, Runtime.getRuntime().availableProcessors()); //Assume majority of hardware have 4 cores at least
    }
    
    /**
     * This method will perform erosion operation on the grayscale image img with given thread number.
     * 
     * @param img The image on which dilation operation is performed
     * @param num_ths Number of threads to start with
     * @return copy of dilated image
     */
    public static MyOwnImage Erosion_grayscale(MyOwnImage img, int num_ths){
        return Erosion_grayscale(img, num_ths, new int[]{1,1,1,1,1,1,1,1,1}, 3);
    }
    
    /**
     * This method will perform erosion operation on the grayscale image img.It will find the maximum value among the pixels that are under the mask [element value 1] and will
 set the origin to the maximum value.
     * 
     * @param img The image on which dilation operation is performed
     * @param num_ths Number of threads to run on
     * @param mask the square mask.
     * @param maskSize the size of the square mask. [i.e., number of rows]
     * @return copy of dilated image
     */
    public static MyOwnImage Erosion_grayscale(MyOwnImage img, int num_ths, int mask[], int maskSize){
        return ExecuteMaskOnImage(img, num_ths, mask, maskSize, MorphOp.Erosion);
    }
}
