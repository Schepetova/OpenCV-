//вызываем картинку
/*#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;


int main()
{
    cv::Mat img = cv::imread("D:/frog.jpg");
    namedWindow("First OpenCV Application", WINDOW_AUTOSIZE);
    cv::imshow("First OpenCV Application", img);
    cv::moveWindow("First OpenCV Application", 0, 45);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
*/


//opencv фильтры, работа с изображением
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;
/*int main()
{
    std::string image_path = samples::findFile("D:/frog.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);
    Mat image = imread(image_path, IMREAD_GRAYSCALE); //делает изображение серым
    imshow("1", image); //вывод изображени€
    Mat out;
    
    img = imread("D:/frog.jpg", 1);

    Mat gray_image;

    cvtColor(img, gray_image, COLOR_BGR2Luv);
    imshow("cvt", gray_image); //мен€ет цветовой тон изображени€

    GaussianBlur(img, out, Size(9, 9), 1.0);
    imshow("BLUR", out); //блюр 

   Rect r(80, 80, 100, 100);
    Mat roi = img(r);
    imshow("roi", roi); //обрезка
    

    Point org(30, 100);
    putText(img, "Text On Image", org,
        FONT_HERSHEY_SCRIPT_COMPLEX, 2.1,
        Scalar(0, 0, 255), 2, LINE_AA);
    imshow("TEXT", img);

    inRange(img, Scalar(0,0,255), Scalar(0,0,255), out); //работа с цветами
    imshow("RANGE",out);


    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
   // imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if (k == 's')
    {
        imwrite("starry_night.png", img);
    }
    return 0;
}

*/


//контуры, круги и линии

/*
int main(int argc, char** argv)
{
    // Declare the output variables
    Mat dst, cdst, cdstP;
    const char* default_file = "D:/frog.jpg";
    const char* filename = argc >= 2 ? argv[1] : default_file;
    // Loads an image
    Mat src = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
    // Check if image is loaded fine
    if (src.empty()) {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", default_file);
        return -1;
    }
    // Edge detection
    Canny(src, dst, 50, 200, 3);
    // Copy edges to the images that will display the results in BGR
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();
    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }
    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        Vec4i l = linesP[i];
        line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }
    // Show results
    imshow("Source", src);
    imshow("Detected Contours - Standard Hough Line Transform", cdst);
    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    const char* filename1 = argc >= 2 ? argv[1] : "D:/circle.jpg";
    // Loads an image
    Mat src1 = imread(samples::findFile(filename1), IMREAD_COLOR);
    // Check if image is loaded fine
    if (src1.empty()) {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", filename1);
        return EXIT_FAILURE;
    }
    Mat gray;
    cvtColor(src1, gray, COLOR_BGR2GRAY);
   // medianBlur(gray, gray, 5); - без него работает лучше (это дл€ качества картинки) 
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
        gray.rows / 16,  // change this value to detect circles with different distances to each other
        100, 30, 1, 100 // change the last two parameters
        // (min_radius & max_radius) to detect larger circles
    );
    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle(src1, center, 1, Scalar(70, 255, 100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle(src1, center, radius, Scalar(70, 255, 100), 3, LINE_AA);
    }
    imshow("detected circles", src1);
    // Wait and Exit
    waitKey();
    return 0;
}
*/

//работа с видео

// Include Libraries
/*#include<opencv2/opencv.hpp>
#include<iostream>

// Namespace to nullify use of cv::function(); syntax
using namespace std;
using namespace cv;

int main()
{
    // initialize a video capture object
    VideoCapture vid_capture("D:/project_video.mp4");

    int iii;
    iii = 0;

    // Print error message if the stream is invalid
    if (!vid_capture.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
    }

    else
    {
        // Obtain fps and frame count by get() method and print
        // You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        int fps = vid_capture.get(5);
        cout << "Frames per second :" << fps;

        // Obtain frame_count using opencv built in frame count reading method
        // You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
        int frame_count = vid_capture.get(7);
        cout << "  Frame count :" << frame_count;
    }


    // Read the frames to the last frame
    while (vid_capture.isOpened())
    {
        // Initialise frame matrix
        Mat frame;

        // Initialize a boolean to check if frames are there or not
        bool isSuccess = vid_capture.read(frame);


        int down_width = 600;
        int down_height = 400;
        Mat resized_down;
        //resize down
        resize(frame, resized_down, Size(down_width, down_height), INTER_LINEAR);
         Mat gray_image;
         Mat out1;
         Mat gausse;
         //Mat gray_image1;
         //Mat gray_image2;
         //Mat gray_image3;



         //cvtColor(resized_down, gray_image, COLOR_BGR2GRAY);
         //cvtColor(resized_down, gray_image, COLOR_BGRA2GRAY);
         cvtColor(resized_down, gray_image, COLOR_BGR2HSV);
         //cvtColor(resized_down, gray_image, COLOR_RGBA2BGR);
  
       
        inRange(gray_image, Scalar(0, 0, 200), Scalar(60, 255, 255), out1);
        //GaussianBlur(out1, gausse, Size(7, 7), 0.0);
       
        Point p1(220, 270), p2(380, 270);
        Point p3(90, 380), p4(545, 380);
        int thickness = 2;

        // Line drawn using 8 connected
        // Bresenham algorithm
        line(resized_down, p1, p2, Scalar(255, 0, 0),
            thickness, LINE_8);

        // Line drawn using 4 connected
        // Bresenham algorithm
        line(resized_down, p1, p3, Scalar(0, 255, 0),
            thickness, LINE_4);

        // Antialiased line
        line(resized_down, p2, p4, Scalar(0, 0, 255),
            thickness, LINE_AA);


        Point2f srcPoints[] = {
            Point(220,270),
            Point(380,270),
            Point(90,380),
            Point(545,380)
        };

        Point2f dstPoints[] = {
        Point(0,0),
        Point(400,0),
        Point(0,400),
        Point(400,400)
        };

        Mat out3;
        Mat Matrix = getPerspectiveTransform(srcPoints, dstPoints);
        warpPerspective(out1, out3, Matrix, Size(400, 400));

        Mat out4;
        warpPerspective(resized_down, out4, Matrix, Size(400, 400)); 

       string stri;
        iii = iii + 1;
        stri = to_string(iii);

        Point org(30, 100);
        putText(resized_down, stri, org,
            FONT_HERSHEY_SCRIPT_COMPLEX, 2.1,
            Scalar(255, 255, 255), 2, LINE_AA);
            
            // Probabilistic Line Transform
        vector<Vec4i> linesP; // will hold the results of the detection
        HoughLinesP(out3, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
        // Draw the lines
        for (size_t i = 0; i < linesP.size(); i++)
        {
            Vec4i l = linesP[i];
            line(out4, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(60, 255, 40), 3, LINE_AA);
        }

        int windows_n_rows = 20;
        int windows_n_cols = 20;

        // Step of each window
        int StepSlide = 10;
        Mat DrawResultGrid = out3.clone();
        vector <cv::Point> tsentry;
        vector <cv::Point> aprox;
        vector <cv::Point> tsentry1;
        vector <cv::Point> aprox1;


        for (int row = 0; row <= out3.rows - windows_n_rows; row += StepSlide)
        {

            for (int col = 0; col <= out3.cols - windows_n_cols; col += StepSlide)
            {

                Rect windows(col, row, windows_n_rows, windows_n_cols);
                Mat Roi = out3(windows);

                //vector<Point2i> locations; // output, locations of non-zero pixels
                //findNonZero(Roi, locations);

                //Mat DrawResultHere = out3.clone();

                // Draw grid
                if (col <= 200 && col>30) {
                    if (cv::countNonZero(Roi)) {
                        rectangle(out4, windows, Scalar(255, 0, 0), 1, 8, 0);
                        tsentry.push_back(Point(col + round(windows_n_cols / 2), row + round(windows_n_rows / 2)));



                        if (tsentry.size() > 100) {


                            approxPolyDP(tsentry, aprox, 40, true);
                            for (size_t i = 0; i < aprox.size() - 1; i++) {
                                line(out4, aprox[i], aprox[i + 1], Scalar(255, 0, 255), 4, LINE_AA);
                            }
                            Point poin = tsentry[0];
                            tsentry.clear();
                            tsentry.push_back(Point(poin));


                        }


                    }
                }
                if (col >200) {
                    if (cv::countNonZero(Roi)) {
                        rectangle(out4, windows, Scalar(255, 0, 0), 1, 8, 0);
                        tsentry1.push_back(Point(col + round(windows_n_cols / 2), row + round(windows_n_rows / 2)));



                        if (tsentry1.size() > 50) {


                            approxPolyDP(tsentry1, aprox1, 40, true);
                            for (size_t i = 0; i < aprox1.size() - 1; i++) {
                                line(out4, aprox1[i], aprox1[i + 1], Scalar(255, 0, 255), 4, LINE_AA);
                            }
                            Point poin = tsentry1[0];
                            tsentry1.clear();
                            tsentry1.push_back(Point(poin));


                        }


                    }
                }
            }
        }
        // If frames are present, show it
        if (isSuccess == true)
        {
            //display frames
            //imshow("Frame", gray_image);
            //imshow("Frame", gausse);
            imshow("Frame1", out1);
            imshow("Frame2", out3);
            //imshow("Frame3", out4);
            imshow("Frame4", resized_down);
            imshow("Frame5", out4);
            //imshow("Frame1", gray_image1);
            //imshow("Frame2", gray_image2);
           
        }

        // If frames are not there, close it
        if (isSuccess == false)
        {
            cout << "Video camera is disconnected" << endl;
            break;
        }

        //wait 20 ms between successive frames and break the loop if key q is pressed
        int key = waitKey(70);
        if (key == 'q')
        {
            cout << "q key is pressed by the user. Stopping the video" << endl;
            break;
        }


    }
    // Release the video capture object
    vid_capture.release();
    destroyAllWindows();
    return 0;
}
*/
// работа с видео 
#include<opencv2/opencv.hpp>
#include<iostream>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;
void MyPolygon(Mat img, Point point1, Point point2, Point point3, Point point4)
{
    int lineType = LINE_8;
    Point rook_points[1][4];
    rook_points[0][0] = point1;
    rook_points[0][1] = point2;
    rook_points[0][2] = point3;
    rook_points[0][3] = point4;
    const Point* ppt[1] = { rook_points[0] };
    int npt[] = { 4 };
    fillPoly(img,
        ppt,
        npt,
        1,
        Scalar(255, 0, 0),
        lineType);
}

int main()
{
    VideoCapture vid_capture("C:/Users/User/OneDrive/–абочий стол/project_video.mp4");
    if (!vid_capture.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
    }
    else
    {
        int fps = vid_capture.get(CAP_PROP_FPS);
        cout << "Frames per second :" << fps;
        int frame_count = vid_capture.get(CAP_PROP_FRAME_COUNT);
        cout << " Frame count :" << frame_count;
    }


    std::vector<cv::Mat> frames;
    int framecount = 0;
    while (vid_capture.isOpened())
    {
        std::vector<cv::Point2i> locations;
        std::vector<cv::Point> points_x;
        std::vector<cv::Point> points_y;
        std::vector<cv::Point> approx_right;
        std::vector<cv::Point> approx_left;
        std::vector<int> points_num_left;
        std::vector<int> points_num_right;
        // Initialise frame matrix. «десь работаем с трапецией. «адаем точки трапеции на видео и мен€ем перспективу дл€ захвата дорожной разметки
        Mat frame;
        Mat frame1;
        Mat frame2;
        bool isSuccess = vid_capture.read(frame);
        if (isSuccess == true)
        {
            framecount += 1;
            inRange(frame, Scalar(0, 0, 0), Scalar(220, 220, 220), frame1);
            inRange(frame1, Scalar(0, 0, 0), Scalar(10, 10, 10), frame1);
            Point A = Point(120, 700);
            Point B = Point(570, 450);
            Point C = Point(750, 450);
            Point D = Point(1200, 700);
            line(frame, A, B, Scalar(250, 250, 0), 2);
            line(frame, B, C, Scalar(250, 250, 0), 2);
            line(frame, C, D, Scalar(250, 250, 0), 2);
            line(frame, D, A, Scalar(250, 250, 0), 2);

            int height = 400;
            int width = 400;
            Point2f outputQuad[4];
            outputQuad[0] = Point2f(0, 0);
            outputQuad[1] = Point2f(0, height);
            outputQuad[2] = Point2f(width, 0);
            outputQuad[3] = Point2f(width, height);

            Mat output_bin, output;
            Point2f inputQuad[4];

            inputQuad[1] = Point2f(120, 700);
            inputQuad[0] = Point2f(570, 450);
            inputQuad[2] = Point2f(750, 450);
            inputQuad[3] = Point2f(1200, 700);

            Mat M = getPerspectiveTransform(inputQuad, outputQuad);
            warpPerspective(frame1, output_bin, M, Size(width, height));
            warpPerspective(frame, output, M, Size(width, height));

            int i, j;
            int k = 20;
            int l = 0;
            int o = 0;
            int x = 0, y = 0;
            frames.push_back(output_bin);
            for (int u = 0; u < frames.size(); u++) {
                for (x = 0; x < width; x += k)
                {
                    for (y = 0; y < height; y += k)
                    {
                        Mat out = frames[u](Rect(x, y, k, k));
                        cv::findNonZero(out, locations);
                        if (locations.size() >= 70)
                        {
                            cv::rectangle(output, Rect(x, y, k, k), cv::Scalar(0, 0, 0));
                            line(frames[u], Point(x + k / 2, y + k / 2), Point(x + k / 2, y + k / 2), Scalar(0, 0, 0), 2, LINE_AA);
                            if (x < width / 2)
                            {
                                points_x.push_back(Point(x + k / 2, y + k / 2));
                                l += 1;
                            }
                            if (x >= width / 2)
                            {
                                o += 1;
                                points_y.push_back(Point(x + k / 2, y + k / 2));
                            }
                        }
                    }
                }
            }
            if (framecount > 90) {
                frames.erase(frames.begin());
            }
            //здесь работаем с заполнением 
            double a = 0;
            double b = 0;
            double sumx = 0;
            double sumy = 0;
            double sumxy = 0;
            double sumx2 = 0;
            int len = points_x.size();
            for (i = 1; i < len; i++) {
                sumx = sumx + points_x[i].x;
                sumxy = sumxy + points_x[i].x * points_x[i].y;
                sumy = sumy + points_x[i].y;
                sumx2 = sumx2 + points_x[i].x * points_x[i].x;
            }
            a = (len * sumxy - sumx * sumy) / (len * sumx2 - sumx * sumx);
            b = (sumy - a * sumx) / len;
            Point point1, point2, point3, point4;
            point1.y = 1;
            point1.x = int((1 - b) / a);
            point2.y = height;
            point2.x = int((height - b) / a);
            sumx = 0;
            sumy = 0;
            sumxy = 0;
            sumx2 = 0;
            line(output, point1, point2, Scalar(255, 0, 255), 10, LINE_AA);
            len = points_y.size();
            for (i = 1; i < o; i++) {
                sumx = sumx + points_y[i].x;
                sumxy =
                    sumxy + points_y[i].y * points_y[i].x;
                sumy = sumy + points_y[i].y;
                sumx2 = sumx2 + points_y[i].x * points_y[i].x;
            }
            a = (o * sumxy - sumx * sumy) / (o * sumx2 - sumx * sumx);
            b = (sumy - a * sumx) / o;
            point3.y = 1;
            point3.x = int((1 - b) / a);
            point4.y = height;
            point4.x = int((height - b) / a);
            line(output, point3, point4, Scalar(255, 0, 255), 10, LINE_AA);

            Mat outputfake = output.clone();
            MyPolygon(outputfake, point2, point1, point3, point4);
            addWeighted(outputfake, 0.50, output, 0.50, 0.0, output);

            M = getPerspectiveTransform(outputQuad, inputQuad);
            Mat frame3 = frame.clone();
            warpPerspective(output, frame3, M, frame.size());
            addWeighted(frame3, 1.0, frame, 1, 0.0, frame);

            imshow("Frame1", output);
            imshow("Frame", frame);
        }
        // If frames are not there, close it
        if (isSuccess == false)
        {
            cout << "Video camera is disconnected" << endl;
            break;
        }
        //wait 20 ms between successive frames and break the loop if key q is pressed
        int key = waitKey(20);
        if (key == 'q')
        {
            cout << "q key is pressed by the user. Stopping the video" << endl;
            break;
        }
    }
    // Release the video capture object
    vid_capture.release();
    destroyAllWindows();
    return 0;
}