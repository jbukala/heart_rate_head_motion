// Author: Hu Yuhuang.
// Date: 20140119
#include "sys_lib.h"
#include "pfhm_lib.h"

int main(void)
{
  PFHM * pfhm=new PFHM("C:\\Users\\i6093682\\Desktop\\pulsefromheadmotion-master\\resources\\haarcascade_frontalface_default.xml"); // 
  cv::VideoCapture capture;

  int Fr = 61;//61; //frame-rate (is 30 for webcam?)
  int secsToProcess = 5; //How many seconds of the video are processed
  int startTime = 5; //how many seconds after the begin of the movie should we start processing (non-functioning now)
  capture.open("C:\\Users\\i6093682\\Desktop\\MAHNOB-HCI\\Subject2\\Sessions\\133\\P2-Rec1-2009.07.10.15.42.44_BW1 _BW_Section_3.avi");
  //capture.open(0);

  string WINDOW_NAME="Pulse from Head Motion";
  string WINDOW_TOTAL_NAME="Total video";
  int window_x=300;
  int window_y=300;
  int MAX_CORNERS = 40;

  cv::namedWindow(WINDOW_NAME.c_str(), cv::WINDOW_AUTOSIZE);
  moveWindow(WINDOW_NAME.c_str(), window_x, window_y);

  cv::namedWindow(WINDOW_TOTAL_NAME.c_str(), cv::WINDOW_AUTOSIZE);
  moveWindow(WINDOW_TOTAL_NAME.c_str(), window_x+300, window_y);

  int counter=0;
  Mat savedFrame;
  Size savedSize;
  vector<vector<Point2f> > corners;

  vector<Point2f> cornerA;
  //Mat cornerA(2, MAX_CORNERS, CV_32FC1);
  while (true)
    {
      Mat frame, buffer, faceFrame;
      if (!capture.isOpened())
	{
	  std::cout << "error" << endl;
	  break;
	}

      capture >> buffer;

      if (buffer.data && counter < Fr * secsToProcess) //only analyze the desired time-frame
	{
	  pfhm->processImage(buffer, frame, 2); // no process image's size
	  Rect face(0,0,10,10);
	  pfhm->findFace(frame, frame, face, faceFrame);

	  imshow(WINDOW_TOTAL_NAME.c_str(), buffer);

	  if (counter==0)
	    {
	      savedFrame=faceFrame;
	      savedSize=savedFrame.size();

		  goodFeaturesToTrack(savedFrame, cornerA, MAX_CORNERS, /*0.05*/0.01, 5.0);
	    }
	  else
	    {
	      vector<Point2f> cornerB;
	      cv::resize(faceFrame, faceFrame, savedSize, 0, 0, INTER_LINEAR);
	      pfhm->performLKFilter(savedFrame, faceFrame, cornerB, 15, MAX_CORNERS);

	      corners.push_back(cornerB);
	    }

	  imshow(WINDOW_NAME.c_str(), faceFrame);
	  counter++;
	  std::cout << counter << endl;
	}
      else break;

      if (cv::waitKey(5)==27)
	{
	  capture.release();
	  cv::destroyWindow(WINDOW_NAME.c_str());
	}
    }

  // release the memory attached to window
  capture.release();
  cv::destroyWindow(WINDOW_NAME.c_str());
  cv::destroyWindow(WINDOW_TOTAL_NAME.c_str());

  Mat data;
  for (int k=0;k<corners.size();k++)
    {    
      Mat curr;
      for (int i=0;i<corners[k].size();i++)
	  {
		curr.push_back(corners[k].at(i).y);
	  }
      std::cout << "Processed :" << k << endl;
	  curr=curr.t(); 
      data.push_back(curr);
    }

  Mat origin=data.t();

  //-------------------------------------------------------------Signal Analysis
  // NOTE: processed is type CV_32F

   ofstream csv("C:\\Users\\i6093682\\Desktop\\rawSignals.csv");
  for(int z=0; z<origin.rows; z++){ //print all processed signals
		  for(int w = 0; w<origin.cols; w++)
		 {
  			  csv << origin.at<float>(z,w) << ", ";
  		 }
		  csv << endl;
	  }
  csv.close();

  //upsampling used to be here

  //std::cout << "Tracked points discarding.." << endl;
  //Mat discarded = Mat::zeros(0, origin.cols, CV_32F);
  //pfhm->discardErraticSignals(origin, discarded);
  //system("pause");
  Mat discarded = origin; //for now skip discarding.. too strict
  
  std::cout << "Applying 5th order Butterworth filter.." << endl;
  //Try impulse input to see freq response/stability:
  //Mat impulseInput = Mat::zeros(1, 1000, CV_32F);
  //impulseInput.at<float>(0,0) = 1;
  //Mat filtered = Mat::zeros(1, 1000, CV_32F);

  Mat filtered = Mat::zeros(discarded.rows, discarded.cols, CV_32F);
  pfhm->applyFilter(discarded, filtered, Fr);
  system("pause");

  std::printf("Upsampling to 250Hz..\n");
  int upSampleRate = 250;
  Mat processed;
  pfhm->interpolation(filtered, processed, Fr, upSampleRate, true);

  //std::cout << "Calc projection.." << endl;
  Mat S;
  //pfhm->calculatePCAProjection(processed.t(), S);
  processed.copyTo(S);

  std::cout << "Selecting best periodic signal.." << endl;
  Mat F;
  S.copyTo(F);
  float heartRate;
  int bestSignal = pfhm->findBestSignal(S, upSampleRate, 20, heartRate);
  Mat Sig=F.row(bestSignal);

  std::printf("Detecting heartbeats..\n");
  double heart_beat;
  pfhm->calculateHeartRate(Sig, heart_beat, upSampleRate, heartRate);

  ofstream csvoutput("C:\\Users\\i6093682\\Desktop\\finalSignals.csv");
  for(int z=0; z<10; z++){ //print all processed signals
		  for(int w = 0; w<F.cols; w++)
		 {
  			  csvoutput << F.at<float>(z,w) << ", ";
  		 }
		  csvoutput << endl;
	  }
  csvoutput.close();

  return 0;
}
