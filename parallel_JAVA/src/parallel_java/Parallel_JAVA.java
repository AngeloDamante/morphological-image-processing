/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package parallel_java;

/**
 *
 * @author Fabian Greavu
 * Special thanks to https://github.com/yusufshakeel/Java-Image-Processing-Project
 * for the sequential version
 */
public class Parallel_JAVA {
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        System.out.println("Executing dilation and erosion on shapes_different.png binary image");
        String image_fn = "../images/examples/shapes_different.png";
        String image_fn_gs = "../images/examples/lena_grayscale.jpg";
        
        String image_fn_out_d = "shapes_different_out_dilated.png";
        String image_fn_out_e = "shapes_different_out_eroded.png";
        
        String image_fn_out_gs_d = "lena_out_dilated.png";
        String image_fn_out_gs_e = "lena_out_eroded.png";
        
        MyOwnImage image_gs = new MyOwnImage();
        image_gs.readImage(image_fn_gs);
        
        MyOwnImage image_bin = new MyOwnImage();
        image_bin.readImage(image_fn);
        
        
        execute_multiple_threads(image_gs, MorphOp.Dilation);
        execute_multiple_threads(image_gs, MorphOp.Erosion);
        
        execute_multiple_threads_binary(image_bin, MorphOp.Dilation);
        execute_multiple_threads_binary(image_bin, MorphOp.Erosion);
        
        execute_one_Thread(image_gs, MorphOp.Dilation);
        execute_one_Thread(image_gs, MorphOp.Erosion);
        
        execute_one_Thread_binary(image_bin, MorphOp.Dilation);
        execute_one_Thread_binary(image_bin, MorphOp.Erosion);
    }
    
    private static void execute_multiple_threads_binary(MyOwnImage image_bin, MorphOp operation){
        long startTime, duration;
        int[] threads_count = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 32, 40, 50, 64, 128, 256, 512, 1024};
        int[] mask = new int[]{1,1,1,1,1,1,1,1,1};
        int maskSize = 3;
        System.out.println("Binary " + (operation == MorphOp.Dilation ? "Dilation:" : "Erosion:"));
        System.out.println("| Threads  | Time (ms)|");
        for (int i = 0; i < threads_count.length; i++) {
            startTime = System.nanoTime();
            if (operation == MorphOp.Dilation)
                Morphology.Dilation_binary(image_bin, true, threads_count[i], mask, maskSize);
            else
                Morphology.Erosion_binary(image_bin, false, threads_count[i], mask, maskSize);
            duration = (System.nanoTime() - startTime);
            System.out.println(String.format("|%10d|%10d|", threads_count[i], duration/1000000));
        }
    }
    
    private static void execute_one_Thread(MyOwnImage image_gs, MorphOp operation){
        long startTime, duration;
        startTime = System.nanoTime();
        System.out.println("Grayscale " + (operation == MorphOp.Dilation ? "Dilation:" : "Erosion:"));
        System.out.println("| Threads  | Time (ms)|");
        if (operation == MorphOp.Dilation)
            Morphology.Dilation_grayscale(image_gs).writeImage("dilated_grayscale_image.png");
        else
            Morphology.Erosion_grayscale(image_gs).writeImage("eroded_grayscale_image.png");
        duration = (System.nanoTime() - startTime);
        System.out.println(String.format("|%10d|%10d|", 1, duration/1000000));
    }
    
    private static void execute_one_Thread_binary(MyOwnImage image_gs, MorphOp operation){
        long startTime, duration;
        startTime = System.nanoTime();
        System.out.println("Binary " + (operation == MorphOp.Dilation ? "Dilation:" : "Erosion:"));
        System.out.println("| Threads  | Time (ms)|");
        if (operation == MorphOp.Dilation)
            Morphology.Dilation_binary(image_gs, true).writeImage("dilated_binary_image.png");
        else
            Morphology.Erosion_binary(image_gs, false).writeImage("eroded_binary_image.png");
        duration = (System.nanoTime() - startTime);
        System.out.println(String.format("|%10d|%10d|", 1, duration/1000000));
    }
    
    private static void execute_multiple_threads(MyOwnImage image_gs, MorphOp operation){
        long startTime, duration;
        int[] threads_count = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 32, 40, 50, 64, 128, 256, 512, 1024};
        int[] mask = new int[]{1,1,1,1,1,1,1,1,1};
        int maskSize = 3;
        System.out.println("Grayscale " + (operation == MorphOp.Dilation ? "Dilation:" : "Erosion:"));
        System.out.println("| Threads  | Time (ms)|");
        for (int i = 0; i < threads_count.length; i++) {
            startTime = System.nanoTime();
            if (operation == MorphOp.Dilation)
                Morphology.Dilation_grayscale(image_gs, threads_count[i], mask, maskSize);
            else
                Morphology.Erosion_grayscale(image_gs, threads_count[i], mask, maskSize);
            duration = (System.nanoTime() - startTime);
            System.out.println(String.format("|%10d|%10d|", threads_count[i], duration/1000000));
        }
    }
    
}
