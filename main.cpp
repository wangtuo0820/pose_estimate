#include <iostream>
#include <opencv/cv.hpp>
#include <queue>
#include <fstream>
#include <time.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

void left_rotate(vector<Point2f>& corners)
{
    Point2f tmp = corners[0];
    for(int i = 0; i < 3; i++)
        corners[i] = corners[i+1];
    corners[3] = tmp;
}

void right_rotate(vector<Point2f>& corners)
{
    Point2f tmp = corners[3];
    for(int i = 3; i > 0; i--)
        corners[i] = corners[i-1];
    corners[0] = tmp;
}

void sort_corners(vector<Point2f> &corners)
{
    // ccw
    Point2f v1 = corners[1] - corners[0];
    Point2f v2 = corners[2] - corners[0];
    double o = (v1.x * v2.y) - (v1.y * v2.x);
    if (o  < 0.0)
        std::swap(corners[1], corners[3]);

    if(cv::norm(corners[1]-corners[0]) > cv::norm(corners[2] - corners[1]))
        right_rotate(corners);
}


int main()
{
    system("clear");
    VideoCapture phone;
    Mat frame;
    const string ip = "http://10.165.11.253:8080/";

    double cameraMatrix_data[3][3] = {
        {485.591189467975, 0, 242.0221986088893},
        {0, 487.9776613728354, 318.5521643095191},
        {0, 0, 1}
    };
    double distCoeffs_data[1][5] = {0.3001847588012539, -1.700429134113373, -0.001673294820417056, -0.0008618394291576169, 2.852349836302627};
    Size rectSize = Size(210, 297);

    Mat cameraMatrix(3,3, CV_64F, cameraMatrix_data);
    Mat distCoeffs(1,5, CV_64F, distCoeffs_data);

    if(!phone.open(ip))
    {
        cout << "Can't Open Camera!" << endl;
        return -1;
    }

    time_t t;
    ofstream outf("./test.txt", ofstream::app);

    namedWindow("demo", WINDOW_KEEPRATIO);
    namedWindow("rect", WINDOW_KEEPRATIO);

    vector<Point3f> worldCoordinate;
    vector<Point2f> imageCoordinate;


    while(true){
        system("clear");
        if(!phone.read(frame)){
            cout << "No Frame" << endl;
            waitKey();
        }

        // rotate and flip frame to right direction
        transpose(frame, frame);
        flip(frame, frame, 1);

        // image preprocess
        Mat gray, bw, rect;
        imwrite("./org.jpg", frame);
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        imwrite("./gray.jpg", gray);
        threshold(gray, bw, 0, 255, THRESH_OTSU);
        bitwise_not(bw, bw);
        imshow("bw", bw);
        imwrite("./bw.jpg", bw);


        morphologyEx(bw, bw, MORPH_OPEN, Mat()); // what on-earth is kernal, does it necessary??

        imshow("open", bw);
        imwrite("./open.jpg", bw);

        morphologyEx(bw, bw, MORPH_CLOSE, Mat());

        imshow("open-close", bw);
        imwrite("./open_close.jpg", bw);


        vector<vector<Point>> contours;
        vector<Point> approxRect;

        vector<vector<Point>> approxRectsSet;

        findContours(bw, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        rect = Mat::zeros(frame.size(), CV_8UC3);

        // find valid poly
        for(int i = 0; i < contours.size(); i++) // find max area of img
        {
            double area = contourArea(contours[i]);
            if(area < 1000)
                continue;

            approxPolyDP(contours[i], approxRect, double(contours[i].size())*0.1, true); // bigger epsilon is, more retangle detected

            if(approxRect.size() != 4)
                continue;

            if(!isContourConvex(approxRect))
                continue;

            // avoid fake rect(border)
#if 1
            double not_zero = 1.0;
            for(auto p : approxRect)
            {
                not_zero *= (p.x*p.y);
            }
            if(not_zero == 0)
            {
                // cout << "ZERO occur!!" << endl;
                continue;
            }
#endif
            approxRectsSet.push_back(approxRect);

            Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
            drawContours(rect, contours, i, color, -1);


        }
        drawContours(rect, approxRectsSet, -1, Scalar::all(255), 3);
        cout << approxRectsSet.size() << " Rects detected" << endl;


        // draw corners and put text
        TermCriteria criteria = TermCriteria(
                    CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,
                    40, //maxCount=40
                    0.001 );  //epsilon=0.001
        for(int i = 0; i < approxRectsSet.size(); i++)
        {
            vector<Point2f> corners;
            approxRect = approxRectsSet[i];
            for(auto p : approxRect)
            {
                circle(rect, p, 4, Scalar(0, 0, 0), 2);
                corners.push_back(p);
            }
            cornerSubPix(gray, corners, Size(5,5), Size(-1,-1), criteria);

            sort_corners(corners);

            int cnt = 0;
            for(int k = 0; k < corners.size(); k++)
            {
                char x_c[8];
                char y_c[8];
                sprintf(x_c, "%.1f", corners[k].x);
                sprintf(y_c, "%.1f", corners[k].y);
                string text = to_string(++cnt) + "("+ string(x_c) + ", "+ string(y_c) + ")";
                
                putText(rect, text, Point(corners[k].x-50, corners[k].y-1), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, Scalar(0,255,0), 1);
            }
            imageCoordinate = corners;  // record image coordinate
        }

        if(approxRectsSet.size() == 1)
        {// assign real world point
            worldCoordinate.push_back(Point3f(-rectSize.width/2, -rectSize.height/2, 0));
            worldCoordinate.push_back(Point3f(+rectSize.width/2, -rectSize.height/2, 0));
            worldCoordinate.push_back(Point3f(+rectSize.width/2, +rectSize.height/2, 0));
            worldCoordinate.push_back(Point3f(-rectSize.width/2, +rectSize.height/2, 0));


            // solvepnp
            Mat tvec, rvec;
            solvePnP(worldCoordinate, imageCoordinate, cameraMatrix, distCoeffs, rvec, tvec);
            //cout << "rvec:\n" << rvec << endl;

            vector<Point2f> reprojectPoints;
            projectPoints(worldCoordinate, rvec, tvec,cameraMatrix, distCoeffs, reprojectPoints);


            double reproject_err = 0.0;
            for(int i = 0; i < 4; i++)
            {
                circle(frame, reprojectPoints[i], 6, Scalar(255,0,0), -1);
                reproject_err += cv::norm(imageCoordinate[i] - reprojectPoints[i]);
            }
            imshow("pro", frame);
            imwrite("./repro.jpg", frame);
            //cout << "Reprojection Error:" << reproject_err/4.0 << endl;

            cv::Mat_<float> rotMat(3, 3);
            rvec.convertTo(rvec, CV_32F);    //旋转向量
            cv::Rodrigues(rvec, rotMat);

            float theta_z = atan2(rotMat[1][0], rotMat[0][0])*57.2958; // 57.2958 -> 180/pi
            float theta_y = atan2(-rotMat[2][0], sqrt(rotMat[2][0] * rotMat[2][0] + rotMat[2][2] * rotMat[2][2]))*57.2958;
            float theta_x = atan2(rotMat[2][1], rotMat[2][2])*57.2958;
#if 0
            cout << "theta_x:" << theta_x << endl;
            cout << "theta_y:" << theta_y << endl;
            cout << "theta_z:" << theta_z << endl;
            cout << "tvec:\n" << tvec << endl;
#endif

            Mat_<float> tMat(3,1);
            tvec.convertTo(tMat, CV_32F);
            t = time(NULL);


            cout << "Attitude:pitch=" << theta_x <<  ",yaw=" << theta_y << ",roll=" << theta_z << endl;
            cout << "Height:" << tMat[2][0] << endl;

            //  cout << "Time:" << t << endl;

            //outf << "Attitude:pitch=" << theta_x <<  ",yaw=" << theta_y << ",roll=" << theta_z << endl;
            // outf << "Height:" << tMat[2][0] << endl;
            //outf << "Time:" << t << endl;

            worldCoordinate.clear();
        }
        imshow("rect", rect);
        imshow("demo", frame);
        if(waitKey(1) == 27)
            break;
    }
    return 0;
}
