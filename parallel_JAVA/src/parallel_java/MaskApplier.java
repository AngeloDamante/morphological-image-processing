/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package parallel_java;
//import java.lang.*;

/**
 *
 * @author Fabian Greavu
 */
public class MaskApplier extends Thread{
    private MyOwnImage img;
    private int output[];
    private int width;
    private int height;
    private int y_min;
    private int y_max;
    
    public MaskApplier(MyOwnImage img, int output[], int y_min, int y_max, int width, int height){
        this.img = img;
        this.output = output;
        this.width = width;
        this.height = height;
        this.y_min = y_min;
        this.y_max = y_max;
    }
    
    @Override
    public void run(){
        ApplyMask();
    }
    
    public void ApplyMask(){
        for(int y = y_min; y < y_max; y++){
            for(int x = 0; x < width; x++){
                //buff
                int buff[];
                buff = new int[9];
                int i = 0;
                for(int ty = y - 1; ty <= y + 1; ty++){
                   for(int tx = x - 1; tx <= x + 1; tx++){
                       if(ty >= 0 && ty < height && tx >= 0 && tx < width){
                           //pixel under the mask
                           buff[i] = img.getRed(tx, ty);
                           i++;
                       }
                   }
                }

                //sort buff
                java.util.Arrays.sort(buff);

                //save highest value
                output[x+y*width] = buff[8];
            }
        }
    }
        
}
