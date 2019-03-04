#include <iostream>
#include <vector>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/calib3d.hpp>
using namespace cv;
using namespace std;

typedef struct typeCalibrateData {
    vector<Point2i> basePoints;
    vector<Point2i> targetPoints;
    Mat baseImage;
    Mat targetImage;
    Mat calibratedImage;
} calibrateData_t;


void baseImageCallback(int32_t event, int32_t x, int32_t y, int32_t flag, void *ustc) {
    vector<Point2i> *pBasePoints = (vector<Point2i> *)ustc;
    if(event == EVENT_FLAG_LBUTTON) {
        cout << "L at " << x << ',' << y << endl;
        (*pBasePoints).push_back(Point2i(x, y));
    }
}

void targetImageCallback(int32_t event, int32_t x, int32_t y, int32_t flag, void *ustc) {
    vector<Point2i> *pTargetPoints = (vector<Point2i> *)ustc;
    if(event == EVENT_FLAG_LBUTTON) {
        cout << "L at " << x << ',' << y << endl;
        (*pTargetPoints).push_back(Point2i(x, y));
    }
}

void labelFinishButtonCallback(int32_t state, void *userdata) {
    auto *pack = (calibrateData_t *)userdata;
    vector<Point2i> pBasePoints = pack->basePoints;
    vector<Point2i> pTargetPoints = pack->targetPoints;
    cout << "Base:" << endl;
    for(uint32_t i = 0; i < pBasePoints.size(); i++) {
        cout << pBasePoints[i] << endl;
    }
    cout << "Target:" << endl;
    for(uint32_t i = 0; i < pTargetPoints.size(); i++) {
        cout << pTargetPoints[i] << endl;
    }
    Mat h = findHomography(pTargetPoints, pBasePoints, RANSAC);
    if(h.rows == 0) {
        cout << "More Points Are Appreciated;" << endl;
        return;
    }
    cout << h << endl;
    vector<Point2f> correctImage;
    vector<Point2f> points;
    vector<Point2f>::iterator iter;
    for(uint32_t row = 0; row < pack->targetImage.rows; row++) {
        for(uint32_t col = 0; col < pack->targetImage.cols; col++) {
            points.push_back(Point2f(col, row));
        }
    }
    perspectiveTransform(points, correctImage, h);
    float minCol, maxCol;
    float minRow, maxRow;
    minCol = maxCol = minRow = maxRow = 0;
    for(iter = correctImage.begin(); iter != correctImage.end(); iter++) {
        if(iter->x < minCol) minCol = iter->x;
        if(iter->x > maxCol) maxCol = iter->x;
        if(iter->y < minRow) minRow = iter->y;
        if(iter->y > maxRow) maxRow = iter->y;
    }
    int32_t sizeRow, sizeCol;
    sizeRow = pack->baseImage.rows;
    sizeCol = pack->baseImage.cols;
    pack->calibratedImage = Mat::zeros(sizeRow, sizeCol, CV_8UC3);
    iter = correctImage.begin();
    for(uint32_t row = 0; row < pack->targetImage.rows; row++) {
        uchar *pSrc = pack->targetImage.ptr<uchar>(row);
        for(uint32_t col = 0; col < pack->targetImage.cols & iter != correctImage.end(); col++) {
            int32_t targetRow = iter->y;
            int32_t targetCol = iter->x;
            if(targetRow < 0 || targetRow >= sizeRow || targetCol < 0 || targetRow >= sizeRow) {
                iter++;
                continue;
            }
            uchar* pTarget = pack->calibratedImage.ptr(targetRow);
            pTarget[targetCol * 3] = pSrc[col * 3];
            pTarget[targetCol * 3 + 1] = pSrc[col * 3 + 1];
            pTarget[targetCol * 3 + 2] = pSrc[col * 3 + 2];
            iter++;
        }
    }
    namedWindow("Calibrated", WINDOW_NORMAL);
    imshow("Calibrated", pack->calibratedImage);
    Mat diff = pack->baseImage - pack->calibratedImage;
    namedWindow("Diff", WINDOW_NORMAL);
    imshow("Diff", diff);
}

int main() {
    calibrateData_t calibratedData;
    Mat baseImage, targetImage, concatImage;
    calibratedData.baseImage = imread("./Image A.jpg", 1);
    calibratedData.targetImage = imread("./Image B.jpg", 1);
    namedWindow("Base Image", WINDOW_NORMAL);
    createButton("Calibrate!", labelFinishButtonCallback, &calibratedData, QT_PUSH_BUTTON);
    imshow("Base Image", calibratedData.baseImage);
    setMouseCallback("Base Image", baseImageCallback, &calibratedData.basePoints);
    namedWindow("Target Image", WINDOW_NORMAL);
    imshow("Target Image", calibratedData.targetImage);
    setMouseCallback("Target Image", targetImageCallback, &calibratedData.targetPoints);
    waitKey(0);
    destroyAllWindows();
    return 0;
}