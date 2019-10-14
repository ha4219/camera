package com.example.opencv;


import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.DialogInterface;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.annotation.TargetApi;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Interpreter.Options.*;
import org.tensorflow.lite.gpu.GpuDelegate;


import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static android.Manifest.permission.CAMERA;


public class MainActivity extends AppCompatActivity
        implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "opencv";
    private Mat matInput;
    private Mat matResult;
    private Bitmap bitmap;
    private TextView mfps = null;
    protected int frameCounter;
    protected long lastNanoTime;

    //resize data
    private int top;
    private int bottom;
    private int left;
    private int right;
    private float ratio;
    private Interpreter.Options options;



    private final int inputSize = 224;
    private int[] intValues = new int[inputSize*inputSize];

    Interpreter interpreter;
    private CameraBridgeViewBase mOpenCvCameraView;

    public native void ConvertRGBtoGray(long matAddrInput, long matAddrResult);


    static {
        System.loadLibrary("opencv_java4");
        System.loadLibrary("native-lib");
    }

    ByteBuffer imgData = ByteBuffer.allocateDirect(
            inputSize * inputSize * 12);

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(0); // front-camera(1),  back-camera(0)
        Log.d("interpreter 실행","interpreter");
//        GpuDelegate delegate = new GpuDelegate();
//        options = (new Interpreter.Options()).addDelegate(delegate);
        interpreter = getTfliteInterpreter("bbs_dog.tflite");
        frameCounter = 0;

        lastNanoTime = System.nanoTime();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "onResume :: Internal OpenCV library not found.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "onResum :: OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }


    public void onDestroy() {
        super.onDestroy();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        fpsCheck();

        matInput = inputFrame.rgba();

        if ( bitmap == null) {
            bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
        }
        if ( matResult == null ) {
            matResult = new Mat(matInput.rows(), matInput.cols(), matInput.type());
            return matResult;
        }
        matInput.copyTo(matResult);

        Core.divide(matInput, matInput, matInput, 255.);
        Mat resize_image = resize_img(matInput, inputSize);
        Utils.matToBitmap(resize_image, bitmap);

        float[][] bbs = predict_bbs();
        Point lt = new Point((int)((bbs[0][0] - left) / ratio), (int)((bbs[0][1] - top) / ratio));
        Point rb = new Point((int)((bbs[0][2] - left) / ratio), (int)((bbs[0][3] - top) / ratio));
//        Log.d(TAG, "width: " + matResult.width());
//        Log.d(TAG, "height: " + matResult.height());
//        Log.d(TAG, "lt x: " + lt.x);
//        Log.d(TAG, "lt y: " + lt.y);
//        Log.d(TAG, "rb x: " + rb.x);
//        Log.d(TAG, "rb y: " + rb.y);
//        Log.d(TAG, "left: " + left);
//        Log.d(TAG, "top: " + top);
//        Log.d(TAG, "ratio: " + ratio);
//        Log.d(TAG, "bbs 0: " + bbs[0][0]);
//        Log.d(TAG, "bbs 1: " + bbs[0][1]);
//        Log.d(TAG, "bbs 2: " + bbs[0][2]);
//        Log.d(TAG, "bbs 3: " + bbs[0][3]);
        Imgproc.rectangle(matResult, lt, rb, new Scalar(255,0,0));
        return matResult;
    }


    public void fpsCheck(){
        frameCounter++;
        final int fps = (int) (frameCounter * 1e9 / (System.nanoTime() - lastNanoTime));
        if(mfps != null){
            Runnable fpsUpdater = new Runnable() {
                @Override
                public void run() {
                    mfps.setText("FPS:" + fps);
                }
            };
            new Handler(Looper.getMainLooper()).post(fpsUpdater);
        }else{
            mfps = (TextView) findViewById(R.id.textView);
        }
//        Log.d(TAG, "fpsCheck: FPS: "+ fps);
    }


    public float[][] predict_bbs(){
        Object[] inputArray = {preprocessing(bitmap)};

        // 결과를 받아올 맵 추가
        // put in result
        float[][] output = new float[][]{new float[4]};
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, output);

        interpreter.runForMultipleInputsOutputs(inputArray, outputs);

        return output;
    }

    public Mat resize_img(Mat img, int scale){
        int old_width = img.width();
        int old_height = img.height();
        ratio = old_width > old_height ? (float) scale / old_width : (float) scale / old_height;
        int new_width = (int) (old_width * ratio);
        int new_height = (int) (old_height * ratio);
        Mat new_image = resize_mat_image(img, new_width, new_height);

        int delta_w = scale - new_image.width();
        int delta_h = scale - new_image.height();

        top = delta_h / 2;
        bottom = delta_h - (delta_h / 2);
        left = delta_w / 2;
        right = delta_w - (delta_w / 2);

        Mat res = new Mat();
        Core.copyMakeBorder(new_image, res, top, bottom, left, right, Core.BORDER_CONSTANT);
        return res;
    }

    public Bitmap resizeBitmapImage(Bitmap source, int maxResolution){
        int width = source.getWidth();
        int height = source.getHeight();
        int newWidth = width;
        int newHeight = height;
        float rate = 0.0f;

        if(width > height){
            if(maxResolution < width){
                rate = maxResolution / (float) width;
                newHeight = (int) (height * rate);
                newWidth = (int ) (width * rate);
            }
        }else{
            if(maxResolution < height){
                rate = maxResolution / (float) height;
                newHeight = (int) (height * rate);
                newWidth = (int ) (width * rate);
            }
        }

        return Bitmap.createScaledBitmap(source, newWidth, newHeight, true);
    }

    public Mat resize_mat_image(Mat source, int width, int height){
        Size size = new Size(width, height);
        Mat res = new Mat();
        Imgproc.resize(source, res, size);

        return res;
    }

    public ByteBuffer preprocessing(Bitmap bitmap){

        // 현재 버퍼의 바이트 순서를 변경
        imgData.order(ByteOrder.nativeOrder());

        // resize
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());

        // position = 0 으로 변경
        imgData.rewind();


        for(int i=0;i<inputSize; ++i){
            for(int j=0;j<inputSize; ++j){
                int pixelValue = -intValues[i * inputSize + j];
                imgData.putFloat((float) (((pixelValue >> 16) & 0xFF)));
                imgData.putFloat((float) (((pixelValue >> 8) & 0xFF)));
                imgData.putFloat((float) ((pixelValue & 0xFF)));
            }
        }

        return imgData;
    }


    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }


    //여기서부턴 퍼미션 관련 메소드
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 200;


    protected void onCameraPermissionGranted() {
        List<? extends CameraBridgeViewBase> cameraViews = getCameraViewList();
        if (cameraViews == null) {
            return;
        }
        for (CameraBridgeViewBase cameraBridgeViewBase: cameraViews) {
            if (cameraBridgeViewBase != null) {
                cameraBridgeViewBase.setCameraPermissionGranted();
            }
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        boolean havePermission = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
                havePermission = false;
            }
        }
        if (havePermission) {
            onCameraPermissionGranted();
        }
    }

    @Override
    @TargetApi(Build.VERSION_CODES.M)
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            onCameraPermissionGranted();
        }else{
            showDialogForPermission("앱을 실행하려면 퍼미션을 허가하셔야합니다.");
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }


    @TargetApi(Build.VERSION_CODES.M)
    private void showDialogForPermission(String msg) {

        AlertDialog.Builder builder = new AlertDialog.Builder( MainActivity.this);
        builder.setTitle("알림");
        builder.setMessage(msg);
        builder.setCancelable(false);
        builder.setPositiveButton("예", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int id){
                requestPermissions(new String[]{CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
            }
        });
        builder.setNegativeButton("아니오", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface arg0, int arg1) {
                finish();
            }
        });
        builder.create().show();
    }


    private Interpreter getTfliteInterpreter(String modelPath) {
        try {
            return new Interpreter(loadModelFile(MainActivity.this, modelPath), options);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private MappedByteBuffer loadModelFile(Activity activity, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}