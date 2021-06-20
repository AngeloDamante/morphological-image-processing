/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package parallel_java;

/**
 *
 * @author malega
 */
public class MyImageTime {
    public MyOwnImage img;
    public float duration;
    
    public MyImageTime(){}
    
    public MyImageTime(MyOwnImage img, float duration){
        this.img = img;
        this.duration = duration;
    }
}
