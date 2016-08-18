#include "pfhm_lib.h"
#include <fstream>
#define NO_CUT 0

//Settings for the algorithms (what steps to include):

#define KMEANS_CUT 0
#define KMEANS_NO_CUT 1
#define KMEANS_TIMESLICE 2
#define NO_KMEANS -1

//Define one of the following to select the kmeans method:
#define KMEANS_METHOD KMEANS_NO_CUT

//discard signals with lot of movement to follow Balakrishnan:
//#define DISCARD 1

//upsample the signal to 250Hz to follow Balakrishnan
//#define UPSAMPLE 1

//Read-in a Matrix from a CSV-file:
Mat readCSV(string filename)
{
	Mat signals = Mat(0,0,CV_32F);
	Mat line = Mat(0,0,CV_32F);
	ifstream file(filename);
	bool looped = false;

	while(file.good()) //read a whole line/signal
	{
		looped = false;
		string s;
		getline(file, s);
		stringstream ss(s);
		string strNum;
		line.release();
		while (getline( ss, strNum, ',' )) //split line/signal into float numbers
		{
			looped = true;
			if (strNum.size() > 1 && !file.eof() ){ //if string isnt empty add the number to it to avoid ',,' string
				line.push_back(stof(strNum));
			}
		}
		if (looped) //to avoid the last (empty) line
		{
			line = line.t();
			signals.push_back(line); //add signal to matrix
		}
	}
	file.close();
	return signals;
}

//calculates entropy of a signal
float calcEntropy(Mat signal) 
{
    /// Establish the number of bins
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
	Mat hist;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    /// Compute the histograms:
    calcHist( &signal, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    hist /= signal.total();

    Mat logP;
    cv::log(hist,logP);

    float entropy = -1*sum(hist.mul(logP)).val[0];

    return entropy;
}

//Use k-means to throw away noisy pieces of the signals and thus return better quality data
//T = timelength in secs of signal-pieces to cluster
Mat kMeansNoiseReduction(Mat data, int Fr, int T)
{
	//First cut up signal in pieces of T secs
	//calc (abs(mean),) std and entro
	Mat pieces;
	
	if (T == NO_CUT) //if T=0 just get whole signals, dont cut
	{
		data.copyTo(pieces);
	}
	else {
		pieces = Mat(data.rows * floor(data.cols / (T*Fr)), T*Fr, CV_32F);
		int rowPtr = 0;
		int colPtr = 0;
		for (int i = 0; i < pieces.rows; i++)
		{
			data(Rect(colPtr, rowPtr, T*Fr, 1)).copyTo(pieces(Rect(0, i, T*Fr, 1)));
			colPtr += T*Fr;
			if (colPtr + T*Fr >= data.cols)
			{
				colPtr = 0;
				rowPtr++;
			}

			//cout << "rowptr, colptr = " << rowPtr << ", " << colPtr << endl;
			//cout << data(Rect(colPtr, rowPtr, T*Fr, 1)) << endl;
		}
	}

	//Calculate features for each piece
	int features = 3;
	Mat samples = Mat(pieces.rows, features, CV_32F);
	Scalar m, stdv = Scalar(0);

	for (int i=0; i<pieces.rows; i++)
	{
		meanStdDev(pieces.row(i), m, stdv);//Calc mean and std of piece
		samples.at<float>(i,0) = abs(m[0]);
		samples.at<float>(i,1) = stdv[0];
		samples.at<float>(i,2) = calcEntropy(pieces.row(i));//Calc entropy of piece
	}

	//Perform actual K-means algorithm
	// http://docs.opencv.org/2.4/modules/core/doc/clustering.html
	int clusters = 3;
	Mat labels;
    int attempts = 5;
    Mat centers;

	//The correct input is cv::Mat inputSamples(numSamples, numFeatures, CV_32F)
	cv::kmeans(samples, clusters, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );

	
	Mat clusterQuality = Mat::zeros(1,clusters, CV_32F);
	for(int i=0; i<clusters; i++)
	{
		clusterQuality.at<float>(0,i) = norm(centers.row(i), NORM_L2); //Select cluster with smallest norm

		for (int j=0; j<pieces.rows; j++) //select cluster with largest amount of points in it
		{
			if (labels.at<int>(j, 0) == i) {clusterQuality.at<float>(0, i) -= 1;}
		}
		
	}

	int minIdx[2];
	minMaxIdx(clusterQuality, NULL, NULL, minIdx);
	int bestCluster = minIdx[1];

	Mat cleanSignals = Mat(0,0, CV_32F);
	for (int i=0; i<pieces.rows;i++)
	{
		if (labels.at<int>(i,0) == bestCluster)
			cleanSignals.push_back(pieces.row(i));
	}
	return cleanSignals;
}

//Use k-means to throw away noisy pieces of the signals and thus return better quality data
//T = timelength in secs of signal-pieces to cluster
//select one time-slice to keep, and keep all signals in this slice.
Mat kMeansNoiseReductionSlice(Mat data, int Fr, int T)
{
	int numSlices = floor((float)data.cols / ((float)T*(float)Fr));
	cout << "This many pieces: " << numSlices << endl;
	//First cut up signal in pieces of T secs
	//calc (abs(mean),) std and entro
	Mat pieces = Mat(data.rows * numSlices, T*Fr, CV_32F);
	Mat timeSlice = Mat(pieces.rows, 1, CV_32F); //to determine in which timeSlice each piece is
	int rowPtr = 0;
	int colPtr = 0;
	int ts = 0; //timeslice
	for (int i = 0; i < pieces.rows; i++)
	{
		timeSlice.at<float>(i, 0) = (float)ts;
		data(Rect(colPtr, rowPtr, T*Fr, 1)).copyTo(pieces(Rect(0, i, T*Fr, 1)));
		colPtr += T*Fr;
		ts++;
		if (colPtr + T*Fr >= data.cols)
		{
			colPtr = 0;
			ts = 0;
			rowPtr++;
		}

		//cout << "rowptr, colptr = " << rowPtr << ", " << colPtr << endl;
		//cout << data(Rect(colPtr, rowPtr, T*Fr, 1)) << endl;
	}

	//Calculate features for each piece
	int features = 3;
	Mat samples = Mat(pieces.rows, features, CV_32F);
	Scalar m, stdv = Scalar(0);

	for (int i = 0; i<pieces.rows; i++)
	{
		meanStdDev(pieces.row(i), m, stdv);//Calc mean and std of piece
		samples.at<float>(i, 0) = abs(m[0]);
		samples.at<float>(i, 1) = stdv[0];
		samples.at<float>(i, 2) = calcEntropy(pieces.row(i));//Calc entropy of piece
	}

	//Perform actual K-means algorithm
	// http://docs.opencv.org/2.4/modules/core/doc/clustering.html
	int clusters = 3;
	Mat labels;
	int attempts = 5;
	Mat centers;

	//The correct input is cv::Mat inputSamples(numSamples, numFeatures, CV_32F)
	cv::kmeans(samples, clusters, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);


	Mat clusterQuality = Mat::zeros(1, clusters, CV_32F);
	for (int i = 0; i<clusters; i++)
	{
		clusterQuality.at<float>(0,i) = norm(centers.row(i), NORM_L2); //Select cluster with smallest norm

		/*
		for (int j = 0; j<pieces.rows; j++) //select cluster with largest amount of points in it
		{
			if (labels.at<int>(j, 0) == i) { clusterQuality.at<float>(0, i) -= 1; }
		}*/

	}

	int minIdx[2];
	minMaxIdx(clusterQuality, NULL, NULL, minIdx);
	int bestCluster = minIdx[1];

	//Find best time-slice:
	int time = 0;
	Mat entryPerSlice = Mat::zeros(numSlices, 1, CV_32F); //no of pieces within the best cluster for each timeslice
	for (int i = 0; i<pieces.rows; i++)
	{
		if (labels.at<int>(i, 0) == bestCluster)
			time = (int)timeSlice.at<float>(i, 0);
			entryPerSlice.at<float>(time,0)++; 
	}

	int maxIdx[2];
	minMaxIdx(entryPerSlice, NULL, NULL, NULL, maxIdx); //best slice := the one with the most signal-pieces in the best cluster
	int bestSlice = maxIdx[0];

	Mat cleanSignals = Mat(0, 0, CV_32F);
	for (int i = 0; i<pieces.rows; i++)
	{
		if ((int)timeSlice.at<float>(i, 0) == bestSlice) //push signal-piece to output if its in the best time-slice
			cleanSignals.push_back(pieces.row(i));
	}
	return cleanSignals;
}

//Read in cleaned signal-pieces from CSV and then start processing them to infer heart rate
void analyzeSignals(PFHM* pfhm, string filename, string outputFolder, float Fr, Mat* data = NULL, int sessionNo = 0) 
{
	  cout << "Framerate is: " << Fr << endl;
	  cout << "Start analyzing signals" << endl;
	  Mat origin;
	  if (data == NULL){ //If no time-series passed as argument, we tracked the points in this video before, so load them from a csv file
		cout << "Reading from CSV: " << outputFolder + filename << endl;
		origin = readCSV(outputFolder + filename);
	  }
	  else //if time-series are passed to the function as an argument (so if we tracked)
	  {
		origin = *data;
	  }

	  //int discardedFirstFrames = Fr*3;
	  //origin = origin(Rect(discardedFirstFrames, 0, origin.cols-discardedFirstFrames, origin.rows)); //discard first frames to give tracker time to settle

	  //only use frames 306 to 2135 as in DCT_landmark and Colour papers:
	  int startFrame = floor(306 * (float)Fr/(float)61);
	  int stopFrame = floor(2135 * (float)Fr/(float)61);
	  if (origin.cols > stopFrame){
		origin = origin(Rect(startFrame, 0, stopFrame - startFrame, origin.rows));
	  }
	  else
	  {
		  origin = origin(Rect(max(0, origin.cols - (stopFrame - startFrame)-1), 0, min(stopFrame - startFrame, origin.cols), origin.rows)); //if signal is too short to do the above, just get last 30 secs
	  }

	  
	  Mat discarded = Mat::zeros(0, origin.cols, CV_32F);
	  #ifdef DISCARD
	  //std::cout << "Discarding erratic tracked points.." << endl;
	  pfhm->discardErraticSignals(origin, discarded, false);
	  #else
	  origin.copyTo(discarded);
	  #endif

	  //cout << "Decimating/downsampling to 30Hz.." << endl;
	  Mat decimated; 
	  //double decimationRate = 30;
	  //pfhm->interpolation(discarded, decimated, Fr, decimationRate, false);
	  discarded.copyTo(decimated); //skip decimation, didnt do any good
	  double decimationRate = Fr;

	  pfhm->meanSignal(decimated, decimated); //center each signal around x-axis (substract mean)

	  std::cout << "Applying 5th order Butterworth filter.." << endl;

	  /* //skip MA filter to replicate balakrishnan
	  cv::Size kernel;
	  kernel.width = 7;
	  kernel.height = 1;
	  blur(decimated, decimated, kernel);
	  */

	  Mat filtered = Mat::zeros(decimated.rows, decimated.cols, CV_32F);
	  pfhm->applyFilter(decimated, filtered, (int)decimationRate, false);

	  int discardedFirstFramesFilter = Fr*2;
	  filtered = filtered(Rect(discardedFirstFramesFilter, 0, filtered.cols-discardedFirstFramesFilter, filtered.rows)); //discard first frames after filtering

	  double upSampleRate = 250;
	  Mat processed;
	  #ifdef UPSAMPLE
	  cout << "Upsampling to 250Hz.." << endl;
	  pfhm->interpolation(filtered, processed, decimationRate, upSampleRate, false);
	  #else	
	  upSampleRate = (double)Fr;
	  filtered.copyTo(processed);
	  #endif

	  
	  #if KMEANS_METHOD==KMEANS_CUT
	  cout << "K-means clustering to eliminate noisy segments..." << endl;
	  processed = kMeansNoiseReduction(processed, (int)Fr, 10); //T=10
	  #elif KMEANS_METHOD==KMEANS_NO_CUT
	  cout << "K-means clustering (no-cut) to eliminate noisy segments..." << endl;
	  processed = kMeansNoiseReduction(processed, (int)Fr, NO_CUT);
	  #elif KMEANS_METHOD==KMEANS_TIMESLICE
	  cout << "K-means clustering for best time-slice to eliminate noisy segments..." << endl;
	  processed = kMeansNoiseReductionSlice(processed, (int)Fr, 10);
	  #endif

	  std::cout << "Calculating PCA-projection.." << endl;
	  Mat S;
	  pfhm->meanSignal(processed, processed); //center each signal around x-axis (substract mean)

	  pfhm->exportSig(processed, outputFolder, "clustered_signals");

	  pfhm->calculatePCAProjection(processed, S, 0.25, false);

	  pfhm->exportSig(S, outputFolder, "PCA_Signals");

	  std::cout << "Selecting best periodic signal.." << endl;
	  Mat F, V;
	  S.copyTo(F);
	  S.copyTo(V);

	  float heartRate = 0;
	  int bestSignal = pfhm->findBestSignalFourier(S, upSampleRate, min(60, S.rows), heartRate, outputFolder, false);
	  //cout << "The best signal according to Fourier is: " << bestSignal << endl;
	  float FFTheartRate = heartRate;

	  bestSignal = pfhm->findBestSignalDCT(V, upSampleRate, min(20, V.rows), heartRate, outputFolder, false);
	  //cout << "The best signal according to DCT is: " << bestSignal << endl;
	  float DCTheartRate = heartRate;

	  //write fftrate and dctrate to csv file (appending it):
	  ofstream exportFile;
	  exportFile.open(outputFolder + "CPP_results.csv", fstream::out | fstream::app);
	  exportFile << to_string(sessionNo) << ", " << to_string(FFTheartRate) << ", " << to_string(DCTheartRate) << endl;;
	  exportFile.close();

	  //This replicates the algorithm used in Balakrishnan to calculate HRV, will not use it in the end:
	  //Mat Sig = F.row(bestSignal);
	  //std::cout << "Detecting heartbeats.." << endl;
	  //double heart_beat;
	  //pfhm->calculateHeartRate(Sig, heart_beat, upSampleRate, heartRate);
}