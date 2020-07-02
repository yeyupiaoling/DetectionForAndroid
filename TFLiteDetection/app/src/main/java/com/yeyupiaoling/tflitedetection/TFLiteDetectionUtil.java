package com.yeyupiaoling.tflitedetection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TFLiteDetectionUtil {
    private static final String TAG = TFLiteDetectionUtil.class.getName();
    private Interpreter tflite;
    private static final int NUM_THREADS = 4;
    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;
    private static final float MIN_SCORE_THRESH = 0.51f;

    private TensorImage inputImageBuffer;
    private ImageProcessor imageProcessor;
    private final int[] imageShape;
    private final TensorBuffer outputLocationBuffer;
    private final TensorBuffer outputClassBuffer;
    private final TensorBuffer outputScoreBuffer;
    private final TensorBuffer outputNumBuffer;


    public TFLiteDetectionUtil(String model_path) throws Exception {
        File file = new File(model_path);
        if (!file.exists()) {
            throw new Exception("model file is not exists!");
        }
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(NUM_THREADS);
            NnApiDelegate delegate = new NnApiDelegate();
//            GpuDelegate delegate = new GpuDelegate();
            options.addDelegate(delegate);
            tflite = new Interpreter(file, options);
            // {1, height, width, 3}
            imageShape = tflite.getInputTensor(tflite.getInputIndex("normalized_input_image_tensor")).shape();
            DataType imageDataType = tflite.getInputTensor(tflite.getInputIndex("normalized_input_image_tensor")).dataType();
            inputImageBuffer = new TensorImage(imageDataType);
            // {1, NUM_DETECTIONS, 4}
            int[] outputLocationsShape = tflite.getOutputTensor(tflite.getOutputIndex("TFLite_Detection_PostProcess")).shape();
            DataType outputLocationsDataType = tflite.getOutputTensor(tflite.getOutputIndex("TFLite_Detection_PostProcess")).dataType();
            outputLocationBuffer = TensorBuffer.createFixedSize(outputLocationsShape, outputLocationsDataType);
            // {1, NUM_CLASSES}
            int[] outputClassesShape = tflite.getOutputTensor(tflite.getOutputIndex("TFLite_Detection_PostProcess:1")).shape();
            DataType outputClassesDataType = tflite.getOutputTensor(tflite.getOutputIndex("TFLite_Detection_PostProcess:1")).dataType();
            outputClassBuffer = TensorBuffer.createFixedSize(outputClassesShape, outputClassesDataType);
            // {1,NUM_CLASSES}
            int[] outputScoresShape = tflite.getOutputTensor(tflite.getOutputIndex("TFLite_Detection_PostProcess:2")).shape();
            DataType outputScoresDataType = tflite.getOutputTensor(tflite.getOutputIndex("TFLite_Detection_PostProcess:2")).dataType();
            outputScoreBuffer = TensorBuffer.createFixedSize(outputScoresShape, outputScoresDataType);
            // {1, 10}
            int[] numDetectionsShape = tflite.getOutputTensor(tflite.getOutputIndex("TFLite_Detection_PostProcess:3")).shape();
            DataType numDetectionsDataType = tflite.getOutputTensor(tflite.getOutputIndex("TFLite_Detection_PostProcess:3")).dataType();
            outputNumBuffer = TensorBuffer.createFixedSize(numDetectionsShape, numDetectionsDataType);
            Log.d(TAG, Arrays.toString(outputLocationsShape));
            Log.d(TAG, Arrays.toString(outputClassesShape));
            Log.d(TAG, Arrays.toString(outputScoresShape));
            Log.d(TAG, Arrays.toString(numDetectionsShape));
        } catch (Exception e) {
            throw new Exception("load model fail!");
        }
        // Creates processor for the TensorImage.
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(imageShape[1], imageShape[2], ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                .build();
    }


    public List<float[]> predictImage(String image_path) throws Exception {
        if (!new File(image_path).exists()) {
            throw new Exception("image file is not exists!");
        }
        FileInputStream fis = new FileInputStream(image_path);
        Bitmap bitmap = BitmapFactory.decodeStream(fis);
        return predictImage(bitmap);
    }


    public List<float[]> predictImage(Bitmap bitmap) throws Exception {
        return predict(bitmap);
    }


    private List<float[]> predict(Bitmap bmp) throws Exception {
        List<float[]> results = new ArrayList<>();
        inputImageBuffer = loadImage(bmp);

        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocationBuffer.getBuffer().rewind());
        outputMap.put(1, outputClassBuffer.getBuffer().rewind());
        outputMap.put(2, outputScoreBuffer.getBuffer().rewind());
        outputMap.put(3, outputNumBuffer.getBuffer().rewind());

        try {
            Object[] inputArray = {inputImageBuffer.getBuffer()};
            tflite.runForMultipleInputsOutputs(inputArray, outputMap);
        } catch (Exception e) {
            throw new Exception("predict image fail!" + e);
        }

        for (int i = 0; i < outputNumBuffer.getIntArray()[0]; i++) {
            if (outputScoreBuffer.getFloatArray()[i] > MIN_SCORE_THRESH) {
                results.add(new float[]{
                        outputLocationBuffer.getFloatArray()[i * 4 + 1],
                        outputLocationBuffer.getFloatArray()[i * 4 + 0],
                        outputLocationBuffer.getFloatArray()[i * 4 + 3],
                        outputLocationBuffer.getFloatArray()[i * 4 + 2],
                        outputClassBuffer.getFloatArray()[i],
                        outputScoreBuffer.getFloatArray()[i]});
            }
        }
        return results;
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);
        return imageProcessor.process(inputImageBuffer);
    }
}
