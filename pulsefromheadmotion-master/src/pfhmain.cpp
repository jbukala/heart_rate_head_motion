#include "sys_lib.h"
#include "continueAnalysis.h"

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <time.h>
#include <Windows.h>
#include <intraface/FaceAlignment.h>
#include <intraface/XXDescriptor.h>

//use only Good Features To Track (Follows Balakrishnan)
#define GFT 1 
//use LandMark points from IntraFace library
#define LM 2 
//use both of the types of points
#define BOTH_POINTS 3

//define one of preceding to select points to track:
#define TRACK_POINTS BOTH_POINTS


bool compareRect(cv::Rect r1, cv::Rect r2) { return r1.height < r2.height; }

int main(int argc, char* argv[])
{
	string videoFile;
	int Fr = 0;
	int sessionNo = 0;
	bool showMovie = 1; //flag whether to show movie & face frames
	bool onlyAnalyze = 0; //flag whether to do tracking or only signal-processing
	if (argc > 1) //when calling from command-line with video-file as argument
	{
		videoFile = argv[1];
		showMovie = 0;

		//CHANGE TO 61 FOR MAHNOB, 30 FOR SELF-MADE:
		//Fr = 61;
		Fr = 30;
		
		if (argc >2) //if given a session number, this is used as an identifier in the cpp_results.csv output. default is 0.
		{
			sessionNo = stoi(argv[2]);
			if (argc > 3)
			{
				onlyAnalyze = 1;
			}
		}
	}
	else
	{
		videoFile = "../example-video/Still90secs75BPM.mp4"; //videofile that is analyzed when .exe is started without arguments
		Fr = 30;
		cout << "Using example video-file." << endl;
		cout << "To use program with your own avi or mp4-file, use:" << endl << "pfhmain.exe [vidFile [vid-id-nr [onlyAnalyze]]]" << endl;
		cout << "Where vidFile is the path to your video file, vid-id-nr is the id under which the results will be saved in output/cpp_results.csv and\
			the onlyAnalyze flag skips the video tracking and attempts to use the trajectories stored in output/rawSignals.csv" << endl;
	}

    string outputFolder = "../output/"; //Folder where all the csv output files are stored

	PFHM * pfhm=new PFHM("../pulsefromheadmotion-master/resources/haarcascade_frontalface_default.xml");
    cv::VideoCapture capture;

	if(onlyAnalyze){
	  #if TRACK_POINTS==GFT
	  analyzeSignals(pfhm, "GFT_signals.csv", outputFolder, float(Fr), NULL, sessionNo); //to replicate Balakrishnan
	  #elif TRACK_POINTS==LM
	  analyzeSignals(pfhm, "landmark_signals.csv", outputFolder, float(Fr), NULL, sessionNo); //using landmark-points only
	  #else
	  analyzeSignals(pfhm, "rawSignals.csv", outputFolder, float(Fr), NULL, sessionNo); //This uses both GFT and SDM points, use one of the lines above to use one of the two instead
	  #endif
	  return 0;
  }

	//INFRAFACE INIT
	char detectionModel[] = "../IntraFace-v1.2-vs2012x32/models/DetectionModel-v1.5.bin";
	char trackingModel[] = "../IntraFace-v1.2-vs2012x32/models/TrackingModel-v1.10.bin";
	string faceDetectionModel("../IntraFace-v1.2-vs2012x32/models/haarcascade_frontalface_alt2.xml");
	// initialize a XXDescriptor object
	INTRAFACE::XXDescriptor xxd(4);
	// initialize a FaceAlignment object
	INTRAFACE::FaceAlignment fa(detectionModel, trackingModel, &xxd);
	if (!fa.Initialized()) {
		cerr << "Intraface: FaceAlignment cannot be initialized." << endl;
		return -1;
	}
	// load OpenCV face detector model
	cv::CascadeClassifier face_cascade;
	if (!face_cascade.load(faceDetectionModel))
	{
		cerr << "Intraface: Error loading face detection model." << endl;
		return -1;
	}

	float score, notFace = 0.3;
	cv::Mat X, X0;
	//END INTRAFACE INIT


    //capture.open(0); //Uncomment this to use webcam as input. Not advised as web-cam framerate was found to vary throughout the recording, with bad results
	if (!capture.open(videoFile))
	{
		cerr << "Error opening video file." << endl;
		return -1;
	}

  //Fr = (int)capture.get(CV_CAP_PROP_FPS);//Let OpenCV detect the framerate of the video automatically (warning: this will not always give the correct framerate)
  int secsToProcess = 90; //How many seconds of the video are processed
  int startTime = 0; //how many seconds after the begin of the movie should we start processing

  cout << "OpenCV Framerate detection: " << capture.get(CV_CAP_PROP_FPS) << endl;
  
  string WINDOW_NAME="Facial region of interest";
  string WINDOW_TOTAL_NAME="Total video";
  if(showMovie){  
	  int window_x=300;
	  int window_y=300;
	  cv::namedWindow(WINDOW_NAME.c_str(), cv::WINDOW_AUTOSIZE);
	  moveWindow(WINDOW_NAME.c_str(), window_x, window_y);
	  cv::namedWindow(WINDOW_TOTAL_NAME.c_str(), cv::WINDOW_AUTOSIZE);
	  moveWindow(WINDOW_TOTAL_NAME.c_str(), window_x+300, window_y);
  }

  int counter=0;
  Mat savedFrame;
  Mat prevFrame;
  Size savedSize;
  Rect face(0,0,10,10);
  vector<vector<Point2f>> corners;
  vector<Point2f> cornerA;
  vector<Point2f> cornerB;

  vector<vector<Point2f>> landmarkPoints;
  vector<Point2f> lm_points_slice;

  int MAX_CORNERS = 100;
  Mat frame, buffer, faceFrame;
  int face_x, face_y, face_height, face_width;

  if (capture.isOpened())
	{
	  for(int i=0;i<Fr*startTime;i++){ //skip to start of desired video-segment
		capture >> buffer;
	  }
	}
  
  cout << "Start tracking.." << endl;
  while (true)
    {
      if (!capture.isOpened())
	{
	  std::cout << "error" << endl;
	  break;
	}

      capture >> buffer;

      if (buffer.data && counter < Fr * secsToProcess) //only analyze the desired time-frame
	{
	   

		if (counter == 0) //First frame of the video: detect a face and determine GFT-points that should be tracked. Also start tracking with IntraFace
		{
			pfhm->processImage(buffer, frame, 2); // no process image's size
			pfhm->findFace(frame, frame, face, faceFrame, face_x, face_y, face_height, face_width);

			if (showMovie) {
				imshow(WINDOW_TOTAL_NAME.c_str(), buffer);
				imshow(WINDOW_NAME.c_str(), faceFrame);
			}

			//IntraFace detect:
			// face detection
			vector<cv::Rect> faces;
			face_cascade.detectMultiScale(frame, faces, 1.2, 2, 0, cv::Size(50, 50));
			// if no face found, do nothing
			if (faces.empty()) {
				continue;
			}
			// facial feature detection on largest face found
			if (fa.Detect(frame, *max_element(faces.begin(), faces.end(), compareRect), X0, score) != INTRAFACE::IF_OK)
				break;
			//END INTRAFACE DETECT

			savedFrame = faceFrame;
			savedSize = savedFrame.size();
			Point2f temp;

			goodFeaturesToTrack(savedFrame, cornerA, MAX_CORNERS, 0.001, 1.0); //Determine good points to track
			for (int i = 0; i < cornerA.size(); i++) //place the corners on the good place of the whole frame
			{
				if (cornerA[i].y > floor((float)face_height*0.308)) //shift points past the eyes-part of the whole-frame
				{
					cornerA[i].y += (float)floor((float)face_height*0.538);
				}
				cornerA[i].y += face_y;
				cornerA[i].x += face_x;
			}
		}
		else
		{
			pfhm->performLKFilter(prevFrame, buffer, cornerA, cornerB, 50); //Use Lukas-Kanade tracking algorithm
			cornerA = cornerB;
			corners.push_back(cornerB); //Save the location of all GFT-points in this frame

			//Intraface:
			//if (fa.Track(frame,X0,X,score) != INTRAFACE::IF_OK)
			//	break;
			fa.Track(frame, X0, X, score);
			X0 = X;

			lm_points_slice.clear(); //empty slice vector
			for (int i = 0; i < X0.cols; i++) {
				lm_points_slice.push_back(Point(X0.at<float>(0, i), X0.at<float>(1, i)));
			}

			landmarkPoints.push_back(lm_points_slice); //Save the location of all LM-points in this frame
		}

	  if(showMovie){ //show the movie and all the tracked points during the tracking process
		  Mat framePoints;
		  buffer.copyTo(framePoints);

		  //draw rectangle around face-region of interest (but still including eye region):
		  //rectangle(framePoints, Rect(face_x, face_y, face_width, face_height), Scalar(255, 0, 0), 2, 8, 0);

		  for(int i=0;i<(int)cornerA.size();i++){ //draw tracked GFT-points on full image
			 circle(framePoints, cornerA[i], 1, Scalar( 0, 0, 255 ), -1, 8);
		  }
		  for(int i=0;i< X0.cols;i++){ //draw tracked landmark-points on full image
			 circle(framePoints, Point(X0.at<float>(0,i), X0.at<float>(1,i)), 1, Scalar( 0, 255, 0 ), -1, 8);
		  }
		  imshow(WINDOW_TOTAL_NAME.c_str(), framePoints); //display images
	  }

	  counter++;
	}
      else break;

      if (cv::waitKey(5)==27)
		{
		  capture.release();
		  cv::destroyWindow(WINDOW_NAME.c_str());
		}
	buffer.copyTo(prevFrame);
    }

  // release the memory attached to window
  capture.release();
  if(showMovie){
	  cv::destroyWindow(WINDOW_NAME.c_str());
	  cv::destroyWindow(WINDOW_TOTAL_NAME.c_str()); 
  }
  
  //Save GFT & LM points separately first and then together as rawsignals
  //First save GFT points
  Mat GFT_data;
  for (int k=0;k<(int)corners.size();k++)
    {    
	  Mat curr;
      for (int i=0;i<(int)corners[k].size();i++)
	  {
		curr.push_back(corners[k].at(i).y);
	  }
	  curr=curr.t(); 
      GFT_data.push_back(curr);
    }
  Mat GFT_origin=GFT_data.t();
  pfhm->exportSig(GFT_origin, outputFolder, "GFT_signals"); //save GFT tracked signals

	//Then save LM points
  Mat LM_data;
  for (int k = 0; k<(int)landmarkPoints.size(); k++)
  {
	  Mat curr;
	  for (int i = 0; i<(int)landmarkPoints[k].size(); i++)
	  {
		  curr.push_back(landmarkPoints[k].at(i).y);
	  }
	  curr = curr.t();
	  LM_data.push_back(curr);
  }
  Mat LM_origin = LM_data.t();
  pfhm->exportSig(LM_origin, outputFolder, "landmark_signals"); //save GFT tracked signals
 

  //----------------------------------------------------Tracking complete - start Signal Analysis
  // NOTE: origin is of type CV_32F

  Mat origin;
  vconcat(GFT_origin, LM_origin, origin);
  pfhm->exportSig(origin, outputFolder, "rawSignals"); //save raw tracked signals

#if TRACK_POINTS==GFT
  analyzeSignals(pfhm, "GFT_signals.csv", outputFolder, float(Fr), &GFT_origin, sessionNo); //to replicate Balakrishnan
#elif TRACK_POINTS==LM
  analyzeSignals(pfhm, "landmark_signals.csv", outputFolder, float(Fr), &LM_origin, sessionNo); //only use LM-points
#else
  analyzeSignals(pfhm, "rawSignals.csv", outputFolder, float(Fr), &origin, sessionNo); //This uses both GFT and SDM points, use one of the lines above to use one of the two instead
#endif

  return 0;
}