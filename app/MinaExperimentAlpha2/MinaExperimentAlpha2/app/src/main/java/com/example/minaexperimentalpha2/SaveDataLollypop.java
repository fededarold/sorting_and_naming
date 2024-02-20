package com.example.minaexperimentalpha2;

import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

class SaveDataLollypop {

    private File myFile;
    FileOutputStream fOut;
    OutputStreamWriter myOutWriter;

    SaveDataLollypop( String fileName ) {
        try {
            File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
            myFile = new File(path, fileName + ".txt");
            fOut = new FileOutputStream(myFile,true);
            myOutWriter = new OutputStreamWriter(fOut);
            //myOutWriter.append("the text I want added to the file");
            //myOutWriter.close();
            //fOut.close();

            //Toast.makeText(this,"Text file Saved !",Toast.LENGTH_LONG).show();
        }

        catch (java.io.IOException e) {

            //do something if an IOException occurs.
            Log.d("SAVE_E", e.toString());
        }
    }

    void append( String s, boolean blank, boolean newline ) {

        char[] c = s.toCharArray();
        for (int i=0; i<c.length; ++i) {
            try {
                myOutWriter.append(c[i]);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        if (blank) {
            char[] b= new char[1];
            b[0] = ' ';
            try {
                myOutWriter.append(b[0]);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        if (newline) {
            char[] b= new char[1];
            b[0] = '\n';
            try {
                myOutWriter.append(b[0]);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    void saveFile() {
        try {
            myOutWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            fOut.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
