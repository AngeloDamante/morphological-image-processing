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
        execute_dil_eros_binary();
    }
    
    private static void execute_dil_eros_binary(){
        long startTime, duration;
        System.out.println("Executing dilation and erosion on shapes_different.png binary image");
        String image_fn = "../images/examples/shapes_different.png";
        String image_fn_gs = "../images/examples/lena_grayscale.jpg";
        
        String image_fn_out_d = "shapes_different_out_dilated.png";
        String image_fn_out_e = "shapes_different_out_eroded.png";
        
        String image_fn_out_gs_d = "lena_out_dilated.png";
        String image_fn_out_gs_e = "lena_out_eroded.png";
        
        //MyOwnImage image = new MyOwnImage();
        //image.readImage(image_fn);
        
        MyOwnImage image_gs = new MyOwnImage();
        image_gs.readImage(image_fn_gs);
        
        //Morphology.Dilation_binary(image, true).writeImage(image_fn_out_d);
        //Morphology.Erosion_binary(image, false).writeImage(image_fn_out_e);
        
        if (false){
            startTime = System.nanoTime();
            Morphology.Dilation_grayscale(image_gs, 1).writeImage(image_fn_out_gs_d);
            duration = (System.nanoTime() - startTime);
            System.out.println(String.format("|%10d|%10d|", 1, duration/1000000));
        }
        
        if (false){
            System.exit(1);//end process here
        }
        System.out.println("| Threads | Time (ms)|");
        int[] threads_count = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 32, 40, 50, 64, 128, 256, 512, 1024};

        for (int i = 0; i < threads_count.length; i++) {
            startTime = System.nanoTime();
            Morphology.Dilation_grayscale(image_gs, threads_count[i]);
            duration = (System.nanoTime() - startTime);
            System.out.println(String.format("|%10d|%10d|", threads_count[i], duration/1000000));
        }
        //Morphology.Dilation_grayscale(image_gs, threads_count[i]).writeImage(image_fn_out_gs_d);
        //Morphology.Erosion_grayscale(image_gs).writeImage(image_fn_out_gs_e);
    }
    
}
