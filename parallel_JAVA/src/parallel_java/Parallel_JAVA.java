/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package parallel_java;

//import java.awt.List;
import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 *
 * @author Fabian Greavu Special thanks to
 * https://github.com/yusufshakeel/Java-Image-Processing-Project for the
 * sequential version
 */
public class Parallel_JAVA {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        System.out.println("Executing dilation and erosion on shapes_different.png binary image");
        //String image_fn_gs = "../images/1280x720/Minecraft_1280x720.png";

        //String image_fn_out_gs_d = "lena_out_dilated.png";
        int[] threads_count = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 32, 64, 128};

        File images_folder = new File("../images/");
        File[] listOfResolutions = images_folder.listFiles();
        Arrays.sort(listOfResolutions);
        List<String> resolutions = new ArrayList<String>();
        List<float[]> durations_total = new ArrayList<>(); //list of resolutions, each one have list of durations per thread 
        MorphOp operation = MorphOp.Dilation;

        for (File folder_resolution : listOfResolutions) {
            resolutions.add(folder_resolution.getName());
            System.out.println(folder_resolution.getName());
            File[] images = folder_resolution.listFiles();

            float[][] durations = new float[images.length][threads_count.length]; //Store every duration
            for (int i = 0; i < images.length; i++) {
                File img = images[i];
                System.out.println(img.getName());
                MyOwnImage image_gs = new MyOwnImage();
                String pth = Paths.get(images_folder.getPath(), folder_resolution.getName(), img.getName()).toString();
                image_gs.readImage(pth);
                long startTime, duration;

                int[] mask = new int[]{1, 1, 1, 1, 1, 1, 1, 1, 1};
                int maskSize = 3;
                System.out.println("Grayscale " + (operation == MorphOp.Dilation ? "Dilation:" : "Erosion:"));
                System.out.println("| Threads  | Time (ms)|");
                for (int j = 0; j < threads_count.length; j++) {
                    startTime = System.nanoTime();
                    if (operation == MorphOp.Dilation) {
                        Morphology.Dilation_grayscale(image_gs, threads_count[j], mask, maskSize);
                    } else {
                        Morphology.Erosion_grayscale(image_gs, threads_count[j], mask, maskSize);
                    }
                    duration = (System.nanoTime() - startTime) / 1000000;
                    durations[i][j] = duration;
                    System.out.println(String.format("|%10d|%10d|", threads_count[j], duration));
                }
            }
            var mean_vals = getAvgFromMatrix(durations);
            durations_total.add(mean_vals);
        }

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
        
        MyCSVWriter.Write("../resuts_java.csv", sb);


        //execute_multiple_threads(image_gs, MorphOp.Dilation);
        //execute_multiple_threads(image_gs, MorphOp.Erosion);
        //execute_multiple_threads_binary(image_bin, MorphOp.Dilation);
        //execute_multiple_threads_binary(image_bin, MorphOp.Erosion);
        //execute_one_Thread(image_gs, MorphOp.Dilation);
        //execute_one_Thread(image_gs, MorphOp.Erosion);
        //execute_one_Thread_binary(image_bin, MorphOp.Dilation);
        //execute_one_Thread_binary(image_bin, MorphOp.Erosion);
    }

    private static float[] getAvgFromMatrix(float[][] durations_total) {
        float[] mean_vals = new float[durations_total[0].length];
        for (int j = 0; j < durations_total[0].length; j++) {
            float sum = 0;
            for (int i = 0; i < durations_total.length; i++) {
                sum += durations_total[i][j];
            }
            mean_vals[j] = sum / durations_total.length;
        }
        return mean_vals;
    }

    private static List<Float> execute_multiple_threads_binary(MyOwnImage image_bin, int[] threads_count, MorphOp operation) {
        long startTime, duration;
        int[] mask = new int[]{1, 1, 1, 1, 1, 1, 1, 1, 1};
        int maskSize = 3;
        System.out.println("Binary " + (operation == MorphOp.Dilation ? "Dilation:" : "Erosion:"));
        System.out.println("| Threads  | Time (ms)|");
        List<Float> durations = new ArrayList<>();
        for (int i = 0; i < threads_count.length; i++) {
            startTime = System.nanoTime();
            if (operation == MorphOp.Dilation) {
                Morphology.Dilation_binary(image_bin, true, threads_count[i], mask, maskSize);
            } else {
                Morphology.Erosion_binary(image_bin, false, threads_count[i], mask, maskSize);
            }
            duration = (System.nanoTime() - startTime);
            durations.add(new Float(duration));
            System.out.println(String.format("|%10d|%10d|", threads_count[i], duration / 1000000));
        }
        return durations;
    }

    private static void execute_one_Thread(MyOwnImage image_gs, MorphOp operation) {
        long startTime, duration;
        startTime = System.nanoTime();
        System.out.println("Grayscale " + (operation == MorphOp.Dilation ? "Dilation:" : "Erosion:"));
        System.out.println("| Threads  | Time (ms)|");
        if (operation == MorphOp.Dilation) {
            Morphology.Dilation_grayscale(image_gs).writeImage("dilated_grayscale_image.png");
        } else {
            Morphology.Erosion_grayscale(image_gs).writeImage("eroded_grayscale_image.png");
        }
        duration = (System.nanoTime() - startTime);
        System.out.println(String.format("|%10d|%10d|", 1, duration / 1000000));
    }

    private static void execute_one_Thread_binary(MyOwnImage image_gs, MorphOp operation) {
        long startTime, duration;
        startTime = System.nanoTime();
        System.out.println("Binary " + (operation == MorphOp.Dilation ? "Dilation:" : "Erosion:"));
        System.out.println("| Threads  | Time (ms)|");
        if (operation == MorphOp.Dilation) {
            Morphology.Dilation_binary(image_gs, true).writeImage("dilated_binary_image.png");
        } else {
            Morphology.Erosion_binary(image_gs, false).writeImage("eroded_binary_image.png");
        }
        duration = (System.nanoTime() - startTime);
        System.out.println(String.format("|%10d|%10d|", 1, duration / 1000000));
    }

    private static List<Float> execute_multiple_threads(MyOwnImage image_gs, int[] threads_count, MorphOp operation) {
        long startTime, duration;
        int[] mask = new int[]{1, 1, 1, 1, 1, 1, 1, 1, 1};
        int maskSize = 3;
        System.out.println("Grayscale " + (operation == MorphOp.Dilation ? "Dilation:" : "Erosion:"));
        System.out.println("| Threads  | Time (ms)|");
        List<Float> durations = new ArrayList<>();
        for (int i = 0; i < threads_count.length; i++) {
            startTime = System.nanoTime();
            if (operation == MorphOp.Dilation) {
                Morphology.Dilation_grayscale(image_gs, threads_count[i], mask, maskSize);
            } else {
                Morphology.Erosion_grayscale(image_gs, threads_count[i], mask, maskSize);
            }
            duration = (System.nanoTime() - startTime);
            durations.add(new Float(duration));
            System.out.println(String.format("|%10d|%10d|", threads_count[i], duration / 1000000));
        }
        return durations;
    }

}
