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
        System.out.println("Executing dilation and erosion on shapes_different.png binary image");
        String image_fn = "../images/examples/shapes_different.png";
        String image_fn_out_d = "shapes_different_out_dilated.png";
        String image_fn_out_e = "shapes_different_out_eroded.png";
        MyOwnImage image = new MyOwnImage();
        image.readImage(image_fn);
        Morphology.Dilation_binary(image, true).writeImage(image_fn_out_d);
        Morphology.Erosion_binary(image, false).writeImage(image_fn_out_e);
    }
    
}
