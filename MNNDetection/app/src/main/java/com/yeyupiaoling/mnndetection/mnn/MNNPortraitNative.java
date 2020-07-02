package com.yeyupiaoling.mnndetection.mnn;

public class MNNPortraitNative {

    public static native int[] nativeConvertMaskToPixelsMultiChannels(float[] mask, int length);

}
