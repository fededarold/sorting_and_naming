package com.example.minaexperimentalpha2;

import android.content.Context;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.support.v7.app.AlertDialog;
import android.text.Editable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;

public class DrawingView extends View {

    protected Paint mPaint;
    protected Bitmap mBitmap;
    protected Canvas mCanvas;

    ArrayList<ArrayList<Integer>> categoryPosition;// = new ArrayList<ArrayList<Integer>>();
    ArrayList<Editable> categoryName = new ArrayList<Editable>();

    //Retrieve the point
    int mx;
    int my;
    int mStartX;
    int mStartY;

    int width;
    int height;

    protected boolean isDrawing;
    boolean haveToDraw;
    boolean isConfirming;
    boolean confirmCategory;

    Context ctx;

    //RelativeLayout mLayout;


    public DrawingView(Context context/*, RelativeLayout cLayout*/) {
        super(context);

        //categoryPosition = new ArrayList<>();

        this.setWillNotDraw(false);
        this.setBackgroundColor(Color.parseColor("#00FF0000"));

        isDrawing = true;
        haveToDraw = true;
        isConfirming = true;

        mPaint = new Paint(Paint.DITHER_FLAG);
        mPaint.setAntiAlias(true);
        mPaint.setDither(true);
        mPaint.setColor(getContext().getResources().getColor(android.R.color.holo_blue_dark));
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeJoin(Paint.Join.ROUND);
        mPaint.setStrokeCap(Paint.Cap.ROUND);
        mPaint.setStrokeWidth(3);



        //mBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        mBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        mCanvas = new Canvas(mBitmap);
        //mCanvas = new Canvas();

        //requestLayout();
        //invalidate();
    }

    public DrawingView(Context context, AttributeSet attrs) {
        super(context, attrs);

        this.setWillNotDraw(false);
        this.setBackgroundColor(Color.parseColor("#00FF0000"));

        //categoryPosition = new ArrayList<>();

        isDrawing = false;

        mPaint = new Paint(Paint.DITHER_FLAG);
        mPaint.setAntiAlias(true);
        mPaint.setDither(true);
        mPaint.setColor(getContext().getResources().getColor(android.R.color.holo_blue_dark));
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeJoin(Paint.Join.ROUND);
        mPaint.setStrokeCap(Paint.Cap.ROUND);
        mPaint.setStrokeWidth(3);

        mCanvas = new Canvas();
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);

        width = w;
        height = h;

        mBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        mCanvas.setBitmap(mBitmap);

        //setCategoriesCoords(4);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        //Log.d("DRAWING", "onDraw");

        //canvas.drawBitmap(mBitmap, 0, 0, mPaint);

        //if (isDrawing && haveToDraw) {

            Log.d("DRAWING", "onDraw in if");

            canvas.drawBitmap(mBitmap, 0, 0, mPaint);

            //drawRectangle(canvas /*, mPaint*/);
            //isDrawing = false;
            //confirmCategory = false;
        //}
    }

    void setCategoriesCoords(int nCategories) {

        int drawingWidth = (int) (width * 0.75);
        int drawingHeight = height;
        int separationMargin = 20;
        int sizeWidth = 0;
        int sizeHeight = 0;
        int numX = 0;
        int numY = 0;

        switch (nCategories) {
            case 2:
                numX = 1;
                numY = 2;
                sizeWidth = drawingWidth - separationMargin;
                sizeHeight = (drawingHeight / numY); // - separationMargin;
                break;
            case 4:
                numX = 2;
                numY = 2;
                sizeWidth = (drawingWidth / numX); // - separationMargin;
                sizeHeight = (drawingHeight / numY); // - separationMargin;
                break;
            case 6:
                numX = 2;
                numY = 3;
                sizeWidth = (drawingWidth / numX); // - separationMargin;
                sizeHeight = (drawingHeight / numY); // - separationMargin;
                break;
            case 10:
                numX = 2;
                numY = 5;
                sizeWidth = (drawingWidth / numX); // - separationMargin;
                sizeHeight = (drawingHeight / numY); // - separationMargin;
                break;
        }

        categoryPosition = setCategoriesEdges(sizeWidth, sizeHeight, numX, numY, separationMargin);
        for (int i = 0; i < nCategories; ++i)
            drawRectangle(categoryPosition.get(i).get(0), categoryPosition.get(i).get(1),
                    categoryPosition.get(i).get(2), categoryPosition.get(i).get(3));

    }

    private ArrayList<ArrayList<Integer>> setCategoriesEdges(int szWidth, int szHeight, int nX, int nY, int margin) {

        ArrayList<ArrayList<Integer>> points = new ArrayList<>();

        for (int i = 0; i < nX; ++i)
            for (int j = 0; j < nY; ++j) {

                ArrayList<Integer> tmpCoords = new ArrayList<>();
                tmpCoords.add((int) (width * 0.25) + (szWidth * i)); //left
                tmpCoords.add(tmpCoords.get(0) + szWidth - margin); // right
                tmpCoords.add((szHeight * j) + margin); //top
                tmpCoords.add(tmpCoords.get(2) + szHeight - (2*margin)); //bottom

                points.add(tmpCoords);
            }

        return points;
    }

    private void drawRectangle(int left, int right, int top, int bottom) {

        Log.d("DRAWING", "drawRectangle");

        mCanvas.drawRect(left, top, right, bottom, mPaint);
        /*if (haveToDraw) {
            handleCategoryDialog(canvas);
        }*/
    }

    public void setCategorising(boolean val) {
        haveToDraw = val;
    }

    public void setContext(Context context) {
        ctx = context;
    }

    void clearCanvas() {
        mCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
    }

    /*private void handleCategoryDialog(final Canvas canvas) {
        isConfirming = true;

        final boolean isCategoryOverlapping;

        final float right = mStartX > mx ? mStartX : mx;
        final float left = mStartX > mx ? mx : mStartX;
        final float bottom = mStartY > my ? mStartY : my;
        final float top = mStartY > my ? my : mStartY;

        // store coords of box, logic is that we store edges coords starting from
        // upper horizontal, bottom h, left v and right v. Easier way to check if boxes overlap
        final ArrayList<Integer> tmpCoords = new ArrayList<>();
        tmpCoords.add((int) left);
        tmpCoords.add((int) right);
        tmpCoords.add((int) top);
        tmpCoords.add((int) bottom);

        isCategoryOverlapping = handleOverlap(tmpCoords);

        AlertDialog.Builder adb = new AlertDialog.Builder(ctx);
        if (isCategoryOverlapping) {
            adb.setTitle("Categories are overlapping.. I cannot create it");
        } else {
            adb.setTitle("Confirm category?");
            //adb.setView(catName);
        }

        adb.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int which) {

                Log.d("DRAWING", "canvas");
                if (!isCategoryOverlapping) {
                    canvas.drawRect(left, top, right, bottom, mPaint);
                    categoryPosition.add(tmpCoords);
                    //categoryName.add(catName.getText());
                }
                //confirmCategory = true;
                invalidate();
                isConfirming = false;
            }
        });
        adb.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int which) {
                //confirmCategory = false;
                isConfirming = false;
            }
        });
        adb.show();
    }

    private boolean handleOverlap(ArrayList<Integer> thisBox) {

        for (int i = 0; i < categoryPosition.size(); ++i) {

            boolean horizontalOverlap = false;
            boolean verticalOverlap = false;

            // intersections
            if (thisBox.get(0) > categoryPosition.get(i).get(0) && thisBox.get(0) < categoryPosition.get(i).get(1)) // < categoryPosition.get(i).get(1))
                horizontalOverlap = true;
            if (thisBox.get(1) > categoryPosition.get(i).get(0) && thisBox.get(1) < categoryPosition.get(i).get(1)) // < categoryPosition.get(i).get(1))
                horizontalOverlap = true;
            if (thisBox.get(2) > categoryPosition.get(i).get(2) && thisBox.get(2) < categoryPosition.get(i).get(3)) // < categoryPosition.get(i).get(1))
                verticalOverlap = true;
            if (thisBox.get(3) > categoryPosition.get(i).get(2) && thisBox.get(3) < categoryPosition.get(i).get(3)) // < categoryPosition.get(i).get(1))
                verticalOverlap = true;
            // inclusion
            if (thisBox.get(0) < categoryPosition.get(i).get(0) && thisBox.get(1) > categoryPosition.get(i).get(1) &&
                    thisBox.get(2) < categoryPosition.get(i).get(2) && thisBox.get(3) > categoryPosition.get(i).get(3)) {
                horizontalOverlap = true;
                verticalOverlap = true;
            }

            if (horizontalOverlap && verticalOverlap)
                return true;
        }
            return false;
    }*/

    ArrayList<ArrayList<Integer>> getCategories() {
        return categoryPosition;
    }
}


    /*@Override
    public boolean onTouchEvent(MotionEvent event) {
        super.onTouchEvent(event);
        Log.d("TOUCH_D", "touchevent");
        //Retrieve the point
        if (!isConfirming && haveToDraw) {

            mx = (int) event.getX();
            my = (int) event.getY();

            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:

                    mStartX = mx;
                    mStartY = my;
                    //invalidate();
                    break;
                case MotionEvent.ACTION_MOVE:
                    //invalidate();
                    break;
                case MotionEvent.ACTION_UP:
                    //Canvas mCanvas = new Canvas();
                    isDrawing = true;

                    Log.d("DRAWING", "action up");

                    //nameCategory();
                    drawRectangle(mCanvas);

                    //invalidate();
                    break;
            }
        }
        return true;
    }*/


        /*private void nameCategory(){
        AlertDialog.Builder adb = new AlertDialog.Builder(ctx);

            adb.setTitle("Please give a name to this category:");
        final EditText catName = new EditText(ctx);
        catName.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        adb.setView(catName);

        adb.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int which) {

                Log.d("DRAWING", "canvas");

                    categoryName.add(catName.getText());
                //drawRectangle(mCanvas);
                invalidate();
            }
        });
        adb.show();
    }*/