package com.yeyupiaoling.tflitedetection.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class CanvasView extends View {
    private static final String TAG = CanvasView.class.getSimpleName();

    private Paint mPaint = new Paint();
    private Paint mTextPaint = new Paint();
    private int mWidth;
    private int mHeight;
    private List<float[]> mList;
    private ArrayList<String> classNames;

    public CanvasView(final Context context, final AttributeSet attrs) {
        super(context, attrs);
        // 文字的画笔
        mTextPaint.setColor(Color.YELLOW);
        mTextPaint.setStyle(Paint.Style.FILL);
        mTextPaint.setAntiAlias(false);
        mTextPaint.setTextSize(36);
        mTextPaint.setFakeBoldText(true);
        // 框的画笔
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setColor(Color.CYAN);
        mPaint.setStrokeWidth(5.0f);
    }

    // 获取View的大小
    public void setTextureViewDimen(int width, int height) {
        mWidth = width;
        mHeight = height;
    }

    // 获取预测数据并进行绘画
    public void populateResultList(List<float[]> list, ArrayList<String> classNames) {
        mList = list;
        this.classNames = classNames;
        postInvalidate();
    }

    @Override
    public void draw(Canvas canvas) {
        super.draw(canvas);
        if (mList == null) {
            canvas.drawColor(Color.TRANSPARENT);
            return;
        }

        for (float[] resultData : mList) {
            canvas.drawRoundRect(resultData[0] * mWidth, resultData[1] * mHeight, resultData[2] * mWidth, resultData[3] * mHeight, 20, 20, mPaint);
            canvas.drawText(classNames.get((int) resultData[4]), resultData[0] * mWidth + 10, resultData[1] * mHeight - 10, mTextPaint);
        }

    }
}
