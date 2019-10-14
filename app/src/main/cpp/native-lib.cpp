#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;

extern "C"
JNIEXPORT void JNICALL
Java_com_example_petshion_MainActivity_ConvertRGBtoGray(JNIEnv *env, jobject instance,
                                                      jlong matAddrInput, jlong matAddrResult) {

    // TODO

    Mat &matInput = *(Mat *)matAddrInput;
    Mat &matResult = *(Mat *)matAddrResult;
    //    cvtColor(matInput, matResult, COLOR_RGB2);
}