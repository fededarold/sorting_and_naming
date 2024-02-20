package com.example.minaexperimentalpha2;

import android.content.Context;
import android.os.Environment;
import android.util.Log;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/** *
 * Created by fede on 31/07/2017.
 *
 *  IO class for saving sensor data on file
 *  The class uses Filewriter and Bufferedwriter *
 *  There are three available data types (int, long, float)
 *
 *  Otherwise, a generic method with String parameter is available
 *  (require data type conversion on method call)
 */

class SnapBackIO {

    //buffer to write file
    private BufferedWriter bW;
    private FileWriter fW;

    //TAG string
    private String TAG = "APP_FILE";

    /**
     * The constructor initialises the file writer and buffer
     * with the given filename. The path refers to the app storage
     * in the SD card.
     * @param context: Activity or Service context to get directories name
     * @param fileName Custom filename
     */
    SnapBackIO(/*Context context,*/ String fileName) {

        //context = context.getApplicationContext();
        File fileSave;

        try {
            /*File[] getDir = context.getExternalFilesDirs(null);
            if (!getDir[1].exists()) {
                if (!getDir[1].mkdirs()) {
                    Log.e(TAG, "directory not created");
                }
            }

            for (int i=0; i<getDir.length; ++i)
                Log.i(TAG, getDir[i].toString());

            //TODO: hardwired for now, [1] is SD card, [0] is virtual storage
            fileSave = new File(getDir[1], fileName + ".txt");*/

            File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
            fileSave = new File(path, fileName + ".txt");

            fW = new FileWriter(fileSave);
            bW = new BufferedWriter(fW);

        } catch (Exception e) {
            Log.e(TAG, e.toString());
        }
    }

    /**
     * Append buffer
     * @param  val: array of FLOAT values
     * @return false if catch exception
     */
    boolean appendFloatToFile(float[] val) {

        StringBuilder sensorValString = new StringBuilder();
        for( int i=0; i<val.length; ++i ) {
            sensorValString.append(val[i] + " ");
        }

        try {
            bW.write(sensorValString.toString());
        } catch (Exception e) {
            Log.e(TAG, e.toString());
            return false;
        }
        return true;
    }

    /**
     * Append buffer
     * @param  val: array of INT  values
     * @return false if catch exception
     */
    boolean appendIntToFile(int[] val) {

        StringBuilder sensorValString = new StringBuilder();
        for( int i=0; i<val.length; ++i ) {
            sensorValString.append(val[i] + " ");
        }

        try {
            bW.write(sensorValString.toString());
        } catch (Exception e) {
            Log.e(TAG, e.toString());
            return false;
        }
        return true;
    }

    /**
     * Append buffer
     * @param  val: array of LONG  values
     * @return false if catch exception
     */
    boolean appendLongToFile(long[] val) {

        StringBuilder sensorValString = new StringBuilder();
        for( int i=0; i<val.length; ++i ) {
            sensorValString.append(val[i] + " ");
        }

        try {
            bW.write(sensorValString.toString());
            //bW.newLine();
        } catch (Exception e) {
            Log.e(TAG, e.toString());
            return false;
        }
        return true;
    }

    /**
     * Append buffer
     * @param  val: String, convert on method call
     * @return false if catch exception
     */
    boolean appendToFile(String val) {

        try {
            bW.write(val + " ");
        } catch (Exception e) {
            Log.e(TAG, e.toString());
            return false;
        }
        return true;
    }

    /**
     * Add newline to buffer
     * @return false if catch exception
     */
    boolean newLine() {
        try {
            bW.newLine();
        } catch (Exception e) {
            Log.e(TAG, e.toString());
            return false;
        }
        return true;
    }

    /**
     * Write file and close
     * @return false if catch exception
     */
    boolean saveFile() {
        try {
            bW.flush();
            bW.close();
            fW.close();
        } catch (IOException e) {
            Log.e(TAG, e.toString());
        }
        return false;
    }

}
