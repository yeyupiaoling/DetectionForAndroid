package com.yeyupiaoling.tflitedetection;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();
    private TFLiteDetectionUtil TFLiteDetectionUtil;
    private ImageView imageView;
    private TextView textView;
    private ArrayList<String> classNames;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        if (!hasPermission()) {
            requestPermission();
        }

        classNames = Utils.ReadListFromFile(getAssets(), "label_list.txt");
        String detectionModelPath = getCacheDir().getAbsolutePath() + File.separator + "ssd_mobilenet_v3_coco.tflite";
        Utils.copyFileFromAsset(MainActivity.this, "ssd_mobilenet_v3_coco.tflite", detectionModelPath);
        try {
            TFLiteDetectionUtil = new TFLiteDetectionUtil(detectionModelPath);
            Log.d(TAG, "模型加载成功");
            Toast.makeText(this, "模型加载成功！", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Log.d(TAG, "模型加载失败", e);
            Toast.makeText(this, "模型加载失败！", Toast.LENGTH_SHORT).show();
        }

        // 获取控件
        Button selectImgBtn = findViewById(R.id.select_img_btn);
        Button openCamera = findViewById(R.id.open_camera);
        imageView = findViewById(R.id.image_view);
        textView = findViewById(R.id.result_text);
        selectImgBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // 打开相册
                Intent intent = new Intent(Intent.ACTION_PICK);
                intent.setType("image/*");
                startActivityForResult(intent, 1);
            }
        });
        openCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // 打开实时拍摄识别页面
                Intent intent = new Intent(MainActivity.this, CameraActivity.class);
                startActivity(intent);
            }
        });
    }



    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == 1) {
                if (data == null) {
                    Log.w("onActivityResult", "user photo data is null");
                    return;
                }
                try {
                    Uri image_uri = data.getData();
                    Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(image_uri));
                    Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    int width = mutableBitmap.getWidth();
                    int height = mutableBitmap.getHeight();
                    int left, top, right, bottom;
                    Canvas canvas = new Canvas(mutableBitmap);
                    Paint paint = new Paint();
                    paint.setColor(Color.GREEN);
                    paint.setStyle(Paint.Style.STROKE);
                    paint.setStrokeWidth(5);
                    Paint paint1 = new Paint();
                    paint1.setColor(Color.BLUE);
                    paint1.setStrokeWidth(10);

                    // 每一个List的值包括 xmin,ymin,xmax,ymax,label,score
                    long start = System.currentTimeMillis();
                    List<float[]> results = TFLiteDetectionUtil.predictImage(bitmap);
                    long end = System.currentTimeMillis();
                    StringBuilder sb = new StringBuilder();
                    for (float[] f : results) {
                        left = (int) (f[0] * width);
                        top = (int) (f[1] * height);
                        right = (int) (f[2] * width);
                        bottom = (int) (f[3] * height);

                        canvas.drawRect(left, top, right, bottom, paint);
                        canvas.drawText(classNames.get((int) f[4]), left, top, paint1);
                        String show_text = "预测结果：" +
                                "\n坐标：("+ left + ", "+ top + ", "+ right + ", "+ bottom + ")" +
                                "\n名称：" +  classNames.get((int) f[4]) +
                                "\n概率：" + f[5] +
                                "\n时间：" + (end - start) + "ms\n";
                        sb.append(show_text);
                    }

                    textView.setText(sb.toString());
                    imageView.setImageBitmap(mutableBitmap);
                    Log.d(TAG, sb.toString());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    // check had permission
    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    // request permission
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{Manifest.permission.CAMERA,
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }
    }
}