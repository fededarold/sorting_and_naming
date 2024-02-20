package com.example.minaexperimentalpha2;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Handler;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.InputType;
import android.text.Layout;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RelativeLayout;
import android.widget.TextClock;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import org.w3c.dom.Text;

import java.util.ArrayList;
import java.util.Random;

enum CurrentWordsType {
    BOTH,
    ABSTRACT,
    CONCRETE
}

public class MainActivity extends AppCompatActivity {

    RelativeLayout mLayout;

    private ArrayList<String> concreteWords = new ArrayList<>();
    private ArrayList<TextView> concreteView = new ArrayList<>();

    private ArrayList<String> abstractWords = new ArrayList<>();
    private ArrayList<TextView> abstractView = new ArrayList<>();

    private ArrayList<String> categoryWords = new ArrayList<>();
    private ArrayList<TextView> categoryView = new ArrayList<>();

    private ArrayList<ArrayList<Integer>> categoryPosition = new ArrayList<>();

    private DrawingView drawingView;

    private Button categoryBtn;
    private Button finishBtn;
    private Handler handlerBtnFinish;

    private final static int TXT_WIDTH = 200;
    private final static int TXT_HEIGHT = 40;
    private final static int NO_CATEGORY = -1;

    private boolean isCategorizing;
    private boolean isStarted;

    private ExperimenterPart experimenterPart;
    private Handler handlerExperimenterPart;

    private static final int REFRESH_UI = 100;

    private boolean concreteFirst;
    private boolean ascendingFirst;
    private int currentNumberOfCategories;
    private int nWords;
    private int currentTrial;
    CurrentWordsType currentWordsType = CurrentWordsType.BOTH;

    SaveDataLollypop saveData;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mLayout = findViewById(R.id.mainlayout);
        isCategorizing = false;

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }

        categoryBtn = findViewById(R.id.btnCategory);
        categoryBtn.setVisibility(View.INVISIBLE);
        handleButtonCategory();
        finishBtn = findViewById(R.id.btnFinish);
        finishBtn.setVisibility(View.INVISIBLE);
        handleButtonFinish();

        handlerBtnFinish = new Handler();

        isStarted = false;
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);

        if (!isStarted) {

            drawingView = findViewById(R.id.cview);
            drawingView.setContext(this);

            experimenterPart = new ExperimenterPart(this, drawingView);
            handlerExperimenterPart = new Handler();
            handlerExperimenterPart.postDelayed(runnableConfigurationDone, REFRESH_UI);
        }

        isStarted = true;
    }

    public final View.OnTouchListener dragCallbackWord = new View.OnTouchListener() {
        @Override
        public boolean onTouch(View v, MotionEvent event) {

            //RelativeLayout.LayoutParams relativeLayoutParams = (RelativeLayout.LayoutParams) v.getLayoutParams();
            if (!isCategorizing) {
                int onPressedX = (int) event.getRawX();
                int onPressedY = (int) event.getRawY();

                Log.d("TOUCH_LAB", "switch out X: " + onPressedX + " Y: " + onPressedY);

                switch (event.getActionMasked()) {
                    case MotionEvent.ACTION_DOWN:
                        onPressedX = (int) event.getRawX();
                        onPressedY = (int) event.getRawY();

                        Log.d("TOUCH_LAB", "switch down: " + onPressedX + " Y: " + onPressedY);
                        break;

                    case MotionEvent.ACTION_MOVE:
                        final int x = (int) event.getRawX();
                        final int y = (int) event.getRawY();

                        // Calculate change in x and change in y
                        int dx = x - onPressedX;
                        int dy = y - onPressedY;

                        Log.d("TOUCH_LAB", "switch move X: " + dx + " Y: " + dy);

                        if (x < mLayout.getWidth() - TXT_WIDTH)
                            v.setX(x);
                        if (y < mLayout.getHeight() - TXT_HEIGHT)
                            v.setY(y);

                        if (isLabelInCategory(x, y) > NO_CATEGORY)
                            v.setBackgroundColor(Color.parseColor("#6699cc00"));
                            //v.setBackgroundColor(getResources().getColor(android.R.color.holo_green_light));
                        else
                            v.setBackgroundColor(Color.parseColor("#00FFFFFF"));
                        //v.setBackgroundColor(getResources().getColor(android.R.color.holo_red_light));

                        break;
                }
            }
            return true;
        }
    };

    public final View.OnTouchListener dragCallbackCategory = new View.OnTouchListener() {
        @Override
        public boolean onTouch(View v, MotionEvent event) {

            //RelativeLayout.LayoutParams relativeLayoutParams = (RelativeLayout.LayoutParams) v.getLayoutParams();
            if (!isCategorizing) {
                int onPressedX = (int) event.getRawX();
                int onPressedY = (int) event.getRawY();

                Log.d("TOUCH_LAB", "switch out X: " + onPressedX + " Y: " + onPressedY);

                switch (event.getActionMasked()) {
                    case MotionEvent.ACTION_DOWN:
                        onPressedX = (int) event.getRawX();
                        onPressedY = (int) event.getRawY();

                        Log.d("TOUCH_LAB", "switch down: " + onPressedX + " Y: " + onPressedY);
                        break;

                    case MotionEvent.ACTION_MOVE:
                        final int x = (int) event.getRawX();
                        final int y = (int) event.getRawY();

                        // Calculate change in x and change in y
                        int dx = x - onPressedX;
                        int dy = y - onPressedY;

                        Log.d("TOUCH_LAB", "switch move X: " + dx + " Y: " + dy);

                        if (x < mLayout.getWidth() - TXT_WIDTH)
                            v.setX(x);
                        if (y < mLayout.getHeight() - TXT_HEIGHT)
                            v.setY(y);

                        if (isLabelInCategory(x, y) > NO_CATEGORY)
                            v.setBackgroundColor(Color.parseColor("#6633b5e5"));
                            //v.setBackgroundColor(getResources().getColor(android.R.color.holo_green_light));
                        else
                            v.setBackgroundColor(Color.parseColor("#D933b5e5"));
                        //v.setBackgroundColor(getResources().getColor(android.R.color.holo_red_light));

                        break;
                }
            }
            return true;
        }
    };

    private void handleButtonFinish() {

        finishBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                AlertDialog.Builder adb = new AlertDialog.Builder(MainActivity.this);
                adb.setTitle("Confirm?");
                adb.setCancelable(false);
                adb.setPositiveButton("YES", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        nextTrial();
                    }
                });
                adb.setNegativeButton("NO", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {

                    }
                });
                adb.show();
            }
        });
    }

    private void nextTrial() {

        finishBtn.setClickable(false);
        // check categories
        if (currentTrial == 1) {
            Log.d("END_BTN", "trial 1");
            saveData = new SaveDataLollypop(experimenterPart.getParticipant());
            saveData.append(experimenterPart.getParticipant(), false, true);
            saveData.append("Trial: " + String.valueOf(currentTrial), false, true);
            saveTrial(saveData, categoryView, "Categories");
            saveTrial(saveData, concreteView, "Concrete");
            saveTrial(saveData, abstractView, "Abstract");
            if (concreteFirst) {
                currentWordsType = CurrentWordsType.CONCRETE;
                deleteTextView(abstractView);
                randomizeTextViewPosition(concreteView, true);
            } else {
                currentWordsType = CurrentWordsType.ABSTRACT;
                deleteTextView(concreteView);
                randomizeTextViewPosition(abstractView, true);
            }

        } else if (currentTrial > 1 && currentTrial <= 5 && concreteFirst) {
            Log.d("END_BTN", "trial concrete");
            saveData.append("Trial: " + String.valueOf(currentTrial), false, true);
            saveTrial(saveData, categoryView, "Categories");
            saveTrial(saveData, concreteView, "Concrete");
            if (currentTrial < 5) {
                currentWordsType = CurrentWordsType.CONCRETE;
                randomizeTextViewPosition(concreteView, true);
            } else {
                currentWordsType = CurrentWordsType.ABSTRACT;
                deleteTextView(concreteView);
                randomizeTextViewPosition(abstractView, false);
            }

        } else if (currentTrial > 5 && currentTrial <= 9 && concreteFirst) {
            Log.d("END_BTN", "trial concrete");
            saveData.append("Trial: " + String.valueOf(currentTrial), false, true);
            saveTrial(saveData, categoryView, "Categories");
            saveTrial(saveData, abstractView, "Abstract");
            randomizeTextViewPosition(abstractView, true);

        } else if (currentTrial > 1 && currentTrial <= 5 && !concreteFirst) {
            Log.d("END_BTN", "trial abstract");
            saveData.append("Trial: " + String.valueOf(currentTrial), false, true);
            saveTrial(saveData, categoryView, "Categories");
            saveTrial(saveData, abstractView, "Abstract");
            if (currentTrial < 5) {
                currentWordsType = CurrentWordsType.ABSTRACT;
                randomizeTextViewPosition(abstractView, true);
            } else {
                currentWordsType = CurrentWordsType.CONCRETE;
                deleteTextView(abstractView);
                randomizeTextViewPosition(concreteView, false);
            }

        } else if (currentTrial > 5 && currentTrial <= 9 && !concreteFirst) {
            Log.d("END_BTN", "trial co");
            saveData.append("Trial: " + String.valueOf(currentTrial), false, true);
            saveTrial(saveData, categoryView, "Categories");
            saveTrial(saveData, concreteView, "Concrete");
            randomizeTextViewPosition(concreteView, true);
        }

        if (currentTrial == 9) {
            deleteTextView(categoryView);
            if (concreteFirst)
                deleteTextView(abstractView);
            else
                deleteTextView(concreteView);
            drawingView.clearCanvas();
            saveData.saveFile();
        } else {
            currentTrial++;
            drawingView.clearCanvas();
            currentNumberOfCategories = estimateCategoriesNumber(currentTrial);
            drawingView.setCategoriesCoords(currentNumberOfCategories);
            finishBtn.setClickable(false);
            for (TextView l : categoryView)
                mLayout.removeView(l);
            categoryView = new ArrayList<>();
            categoryWords = new ArrayList<>();
            handlerBtnFinish.postDelayed(runnableBtnFinish, REFRESH_UI);
            categoryPosition = drawingView.getCategories();
        }
    }


    private int estimateCategoriesNumber(int trial) {

        int[] nCat = {2, 4, 6, 10};
        trial--; // exclude first trial
        if (trial > 4) // exclude 4 trials
            trial = trial - 4;
        if (ascendingFirst)
            return nCat[trial - 1];
        else
            return nCat[4 - trial];
    }

    private void saveTrial(SaveDataLollypop saver, ArrayList<TextView> txt, String info) {
        saver.append(info, false, true);
        for (TextView a : txt) {
            saver.append(String.valueOf(a.getText()), true, false);
            saver.append(String.valueOf(isLabelInCategory((int) a.getX(), (int) a.getY())), false, true);
        }
    }

    private void handleButtonCategory() {

        categoryBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                categoryBtn.setClickable(false);
                categoryView.add(new TextView(MainActivity.this));
                createCategoryLabel(categoryView.get(categoryView.size() - 1));
                AlertDialog.Builder adb = new AlertDialog.Builder(MainActivity.this);
                final EditText inputCategory = new EditText(MainActivity.this);
                inputCategory.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
                adb.setView(inputCategory);
                adb.setCancelable(false);
                adb.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        categoryWords.add(String.valueOf(inputCategory.getText()));
                        categoryView.get(categoryWords.size() - 1).setText(inputCategory.getText());
                        categoryBtn.setClickable(true);
                    }
                });
                adb.show();

            }
        });
    }

    private int isLabelInCategory(int w, int h) {

        Log.d("LABEL_POS", w + "  " + h);

        for (int i = 0; i < categoryPosition.size(); ++i) {

            if (w > categoryPosition.get(i).get(0) && w < categoryPosition.get(i).get(1) - TXT_WIDTH &&
                    h > categoryPosition.get(i).get(2) && h < categoryPosition.get(i).get(3) - TXT_HEIGHT)
                return i;
        }
        return NO_CATEGORY;
    }

    private void randomizeTextViewPosition(ArrayList<TextView> txt, boolean isAlreadyShown) {
        int width = mLayout.getWidth();
        int height = mLayout.getHeight();

        for (TextView v : txt) {
            if (!isAlreadyShown)
                mLayout.addView(v);

            v.setX(new Random().nextInt(width / 5));
            v.setY(new Random().nextInt(height / 2) + height / 4);
            v.setBackgroundColor(Color.parseColor("#00FFFFFF"));
        }
    }

    private void randomizeWordsPosition(ArrayList<String> words, ArrayList<TextView> txt) {

        for (int i = 0; i < words.size(); ++i) {
            txt.add(new TextView(this));
            txt.get(i).setLayoutParams(new RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.WRAP_CONTENT,
                    RelativeLayout.LayoutParams.WRAP_CONTENT));
            txt.get(i).setText(words.get(i));

            mLayout.addView(txt.get(i));
        }

        int width = mLayout.getWidth();
        int height = mLayout.getHeight();

        for (TextView v : txt) {
            //v.setX(new Random().nextInt(width / 2) + width / 4);
            //v.setY(new Random().nextInt(height / 2) + height / 4);
            v.setX(new Random().nextInt(width / 5));
            v.setY(new Random().nextInt(height / 2) + height / 4);
            v.setTextSize(14);
            v.setWidth(TXT_WIDTH);
            v.setHeight(TXT_HEIGHT);
            //v.setTypeface(null, Typeface.BOLD);
            v.setGravity(Gravity.CENTER_VERTICAL | Gravity.CENTER_HORIZONTAL);
            v.setBackgroundColor(Color.parseColor("#00FFFFFF"));
            //v.setAlpha(0.5f);
            v.setOnTouchListener(dragCallbackWord);
        }
    }

    private void deleteTextView(ArrayList<TextView> txt) {

        for (TextView v : txt)
            mLayout.removeView(v);
    }

    private boolean endTrialButtonCheck(CurrentWordsType type) {
        boolean flag = true;

        if (categoryView.size() < currentNumberOfCategories)
            flag = false;

        for (TextView l : categoryView)
            if (isLabelInCategory((int) l.getX(), (int) l.getY()) == NO_CATEGORY)
                flag = false;

        if (type == CurrentWordsType.BOTH || type == CurrentWordsType.CONCRETE)
            for (TextView c : concreteView)
                if (isLabelInCategory((int) c.getX(), (int) c.getY()) == NO_CATEGORY)
                    flag = false;

        if (type == CurrentWordsType.BOTH || type == CurrentWordsType.ABSTRACT)
            for (TextView a : abstractView)
                if (isLabelInCategory((int) a.getX(), (int) a.getY()) == NO_CATEGORY)
                    flag = false;

        if (currentNumberOfCategories == 10)
            flag = true;

        return flag;
    }

    private void createCategoryLabel(TextView c) {

        c.setLayoutParams(new RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.WRAP_CONTENT,
                RelativeLayout.LayoutParams.WRAP_CONTENT));
        mLayout.addView(c);

        c.setTextSize(18);
        c.setWidth(TXT_WIDTH);
        c.setHeight(TXT_HEIGHT);
        c.setTypeface(null, Typeface.BOLD);
        c.setGravity(Gravity.CENTER_VERTICAL | Gravity.CENTER_HORIZONTAL);
        c.setBackgroundColor(Color.parseColor("#D933b5e5"));
        //c.setBackgroundColor(getResources().getColor(android.R.color.holo_blue_bright));
        c.setX(80);
        c.setY(mLayout.getHeight() / 2);
        //v.setAlpha(0.5f);
        c.setOnTouchListener(dragCallbackCategory);
    }

    private Runnable runnableConfigurationDone = new Runnable() {
        @Override
        public void run() {

            Log.d("CHECK_FLAG", "conf: " + experimenterPart.isConfigurationDone());
            if (experimenterPart.isConfigurationDone()) {
                handlerBtnFinish.postDelayed(runnableBtnFinish, REFRESH_UI);
                nWords = experimenterPart.getnWords();
                concreteWords = experimenterPart.getConcreteWords();
                abstractWords = experimenterPart.getAbstractWords();
                randomizeWordsPosition(concreteWords, concreteView);
                randomizeWordsPosition(abstractWords, abstractView);
                finishBtn.setVisibility(View.VISIBLE);
                finishBtn.setClickable(false);
                categoryBtn.setVisibility(View.VISIBLE);
                concreteFirst = experimenterPart.isConcreteFirst();
                ascendingFirst = experimenterPart.isAscendingFirst();
                currentTrial = 1;
                currentNumberOfCategories = 2;
                drawingView.setCategoriesCoords(2);
            }
            if (drawingView.getCategories() != null) {
                categoryPosition = drawingView.getCategories();
                handlerExperimenterPart.removeCallbacks(runnableConfigurationDone);
            } else {
                handlerExperimenterPart.postDelayed(runnableConfigurationDone, REFRESH_UI);
            }
        }
    };

    private Runnable runnableBtnFinish = new Runnable() {
        @Override
        public void run() {

            Log.d("CHECK_BTN", "label: " + categoryView.size() + " cCat " + currentNumberOfCategories + " flag: " + endTrialButtonCheck(currentWordsType) + " type " + currentWordsType);

            if (categoryView.size() >= currentNumberOfCategories || currentNumberOfCategories == 10 && endTrialButtonCheck(currentWordsType)) {
                finishBtn.setClickable(true);
                handlerBtnFinish.removeCallbacks(runnableBtnFinish);
            } else {
                handlerBtnFinish.postDelayed(runnableBtnFinish, REFRESH_UI);
            }
        }
    };

    @Override
    public void onBackPressed() {

    }
}


/*************************************
 *
 *************************************/

