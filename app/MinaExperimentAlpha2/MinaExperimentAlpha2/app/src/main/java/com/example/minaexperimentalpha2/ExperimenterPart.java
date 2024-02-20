package com.example.minaexperimentalpha2;

import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.support.v7.app.AlertDialog;
import android.text.InputType;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.ToggleButton;

import java.util.ArrayList;

class ExperimenterPart {

    private Context context;

    private TextView abstractLabel, abstractList, concreteLabel, concreteList;
    private RadioGroup concreteOrAbstract, ascendingOrDescending;
    private Button confDoneBtn;

    private ArrayList<String> concreteArray = new ArrayList<>();
    private ArrayList<String> abstractArray = new ArrayList<>();
    private String participant;
    private boolean concreteFirst;
    private boolean ascendingFirst;

    DrawingView drawingView;

    private int nWords;
    private boolean isDone;

    private static final String SHARED_PREFS_CONCRETE  = "concrete";
    private static final String SHARED_PREFS_ABSTRACT  = "abstract";

    private static final int INFINITE = 10;

    ExperimenterPart(Context context, DrawingView dw) {
        this.context = context;
        drawingView = dw;

        isDone = false;

        abstractLabel = ((Activity) context).findViewById(R.id.abstract_label);
        abstractList = ((Activity) context).findViewById(R.id.abstract_list);
        concreteLabel = ((Activity) context).findViewById(R.id.concrete_label);
        concreteList = ((Activity) context).findViewById(R.id.concrete_list);

        concreteOrAbstract = ((Activity) context).findViewById(R.id.concrete_or_abstract_group);
        ascendingOrDescending = ((Activity) context).findViewById(R.id.ascending_or_descending_group);
        confDoneBtn = ((Activity) context).findViewById(R.id.btnTrial);

        concreteFirst = true;
        ascendingFirst = true;

        handleTrialSetup();
    }

    private void handleTrialSetup() {

        confDoneBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                loadSavedWords();
                concreteOrAbstract.setVisibility(View.INVISIBLE);
                ascendingOrDescending.setVisibility(View.INVISIBLE);
                confDoneBtn.setVisibility(View.INVISIBLE);
            }
        });

        concreteOrAbstract.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                if (checkedId == R.id.concrete_first)
                    concreteFirst = true;
                else
                    concreteFirst = false;
            }
        });

        ascendingOrDescending.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                if (checkedId == R.id.ascending_first)
                    ascendingFirst = true;
                else
                    ascendingFirst = false;
            }
        });
    }

    private void loadSavedWords() {

        AlertDialog.Builder adb = new AlertDialog.Builder(context);
        adb.setTitle("Load saved words?");
        adb.setCancelable(false);
        adb.setPositiveButton("YES", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int which) {
                concreteArray = getWordsFromSharedPrefs(SHARED_PREFS_CONCRETE);
                abstractArray = getWordsFromSharedPrefs(SHARED_PREFS_ABSTRACT);
                setParticipant();
            }
        });
        adb.setNegativeButton("NO", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                numberOfWords();
            }
        });
        adb.show();
    }

    private void numberOfWords() {

        AlertDialog.Builder adb = new AlertDialog.Builder(context);
        adb.setTitle("Number of words for each class (e.g. 20 = 20 concrete + 20 abstract):");
        final EditText nWordsInput = new EditText(context);
        nWordsInput.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        adb.setView(nWordsInput);
        adb.setCancelable(false);
        adb.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int which) {
                nWords = Integer.valueOf(String.valueOf(nWordsInput.getText()));
                inputConcreteWords();
            }
        });
        adb.show();
    }

    private void inputConcreteWords() {

        AlertDialog.Builder adb = new AlertDialog.Builder(context);
        adb.setTitle("CONCRETE:");
        final EditText nWordsInput = new EditText(context);
        nWordsInput.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        adb.setView(nWordsInput);
        adb.setCancelable(false);
        adb.setNeutralButton("Add", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                concreteArray.add(String.valueOf(nWordsInput.getText()));
                //concreteArray.add("\n");
                String tmpString = "";
                for (String s : concreteArray) {
                    tmpString += s;
                    tmpString += "\n";
                }
                concreteList.setText(tmpString);
                if (concreteArray.size() < nWords) {
                    inputConcreteWords();
                } else {
                    addWordsToSharedPrefs(SHARED_PREFS_CONCRETE, concreteArray);
                    inputAbstractWords();
                }

            }
        });
        adb.show();
    }

    private void inputAbstractWords() {

        AlertDialog.Builder adb = new AlertDialog.Builder(context);
        adb.setTitle("ABSTRACT:");
        final EditText nWordsInput = new EditText(context);
        nWordsInput.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        adb.setView(nWordsInput);
        adb.setCancelable(false);
        adb.setNeutralButton("Add", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                abstractArray.add(String.valueOf(nWordsInput.getText()));
                //abstractArray.add("\n");

                String tmpString = "";
                for (String s : abstractArray) {
                    tmpString += s;
                    tmpString += "\n";
                }

                abstractList.setText(tmpString);
                if (abstractArray.size() < nWords) {
                    inputAbstractWords();
                } else {
                    addWordsToSharedPrefs(SHARED_PREFS_ABSTRACT, abstractArray);
                    setParticipant();
                }
            }
        });
        adb.show();
    }

    private void setParticipant() {

        AlertDialog.Builder adb = new AlertDialog.Builder(context);
        adb.setTitle("Participant");
        final EditText inputParticipant = new EditText(context);
        inputParticipant.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        adb.setView(inputParticipant);
        adb.setCancelable(false);
        adb.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int which) {
                participant = String.valueOf(inputParticipant.getText());
                isDone = true;
                changeUI();
            }
        });
        adb.show();
    }

    private void changeUI() {

        concreteLabel.setVisibility(View.INVISIBLE);
        concreteList.setVisibility(View.INVISIBLE);
        abstractLabel.setVisibility(View.INVISIBLE);
        abstractList.setVisibility(View.INVISIBLE);

        concreteOrAbstract.setVisibility(View.INVISIBLE);
        ascendingOrDescending.setVisibility(View.INVISIBLE);
        confDoneBtn.setVisibility(View.INVISIBLE);

        /*if (ascendingFirst)
            drawingView.setCategoriesCoords(2);
        else
            drawingView.setCategoriesCoords(INFINITE);*/

    }

    private void addWordsToSharedPrefs( String label, ArrayList<String> words ) {

        final SharedPreferences sh = context.getSharedPreferences("stored_words", Context.MODE_PRIVATE);
        final SharedPreferences.Editor ed = sh.edit();
        // tidy up
        int size = sh.getInt(label + "_size", 0);
        for (int i = 0; i <= size; i++) {
            ed.remove(label + "_" + i);
        }
        // write
        ed.putInt(label + "_size", words.size());
        for (int i = 0; i < words.size(); i++) {
            ed.putString(label + "_" + i, words.get(i));
        }
        ed.apply();
    }

    private ArrayList<String> getWordsFromSharedPrefs( String label ) {

        final SharedPreferences sh = context.getSharedPreferences("stored_words", Context.MODE_PRIVATE);
        int size = sh.getInt(label + "_size", 0);
        ArrayList<String> words = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            words.add(sh.getString(label + "_" + i, null));
        }

        return words;
    }

    boolean isConfigurationDone() {
        return isDone;
    }

    ArrayList<String> getConcreteWords() {
        return concreteArray;
    }

    ArrayList<String> getAbstractWords() {
        return abstractArray;
    }

    String getParticipant() {
        return participant;
    }

    boolean isAscendingFirst() {
        return ascendingFirst;
    }

    boolean isConcreteFirst() {
        return concreteFirst;
    }

    int getnWords() {
        return nWords;
    }
}
