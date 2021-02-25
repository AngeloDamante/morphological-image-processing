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
public class BinaryMaskApplier extends Thread{
    private MyOwnImage img;
    private int output[];
    private int width;
    private int height;
    private int y_min;
    private int y_max;
    private int mask[];
    private int maskSize;
    private MorphOp operation;//True if Dilation, false if erosion
    private int targetValue;
    private int reverseValue;
    
    public BinaryMaskApplier(MyOwnImage img, int output[], int y_min, int y_max, int width, int height, int mask[], int maskSize, MorphOp operation, boolean ExecuteOnBackgroundPixel){
        this.img = img;
        this.output = output;
        this.width = width;
        this.height = height;
        this.y_min = y_min;
        this.y_max = y_max;
        //TODO: insert future check on square matrix and odd
        this.mask = mask;
        this.maskSize = maskSize;
        this.operation=operation;
        this.targetValue = (ExecuteOnBackgroundPixel)?0:255;
        this.reverseValue = (this.targetValue == 255)?0:255;
    }
    
    @Override
    public void run(){
        ApplyMaskBinary();
    }
    
    public void ApplyMaskBinary(){
        for(int y = y_min; y < y_max; y++){
            for(int x = 0; x < width; x++){
                if(img.getRed(x, y) == targetValue){
                    boolean flag = false;   //this will be set if a pixel of reverse value is found in the mask
                    for(int ty = y - maskSize/2, mr = 0; ty <= y + maskSize/2; ty++, mr++){
                        for(int tx = x - maskSize/2, mc = 0; tx <= x + maskSize/2; tx++, mc++){
                            if(ty >= 0 && ty < height && tx >= 0 && tx < width){
                                //origin of the mask is on the image pixels
                                if(img.getRed(tx, ty) != targetValue){
                                    flag = true;
                                    output[x+y*width] = reverseValue;
                                }
                            }
                        }
                    }
                    if(!flag){
                        //all pixels inside the mask [i.e., kernel] were of targetValue
                        output[x+y*width] = targetValue;
                    }
                }else{
                    output[x+y*width] = reverseValue;
                }
            }
        }
    }
        
}
