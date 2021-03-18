/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package parallel_java;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
/**
 *
 * @author malega
 */
public class MyCSVWriter {
    public static void Write(String filename, StringBuilder sb) {
        try (PrintWriter writer = new PrintWriter(new File(filename))) {
            writer.write(sb.toString());
            System.out.println("CSV written!");
        } catch (FileNotFoundException e) {
            System.out.println(e.getMessage());
        }
  }
}
