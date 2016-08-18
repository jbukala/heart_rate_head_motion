#include "sys_lib.h"
#define PI 3.14159265

class PFHM
{
 private:
  string fn_haar;
  CascadeClassifier haar_cascade;

 public:
  // Constructor: to load haar features
  // path: absolute path to haar features
  PFHM(string path)
    {
      fn_haar=path;
      if (!haar_cascade.load(fn_haar) )
	  { 
		  printf("--(!)Error loading xml\n");
	  };
    }

  // Function: resize image to desire size
  // in: input image
  // out: resized image
  // mode: specify process mode
  //       PROCESSIMAGE=1;
  //       NOPROCESSIMAGE=2;
  void processImage(Mat in, Mat & out, int mode)
  {
    const int PROCESSIMAGE=1;
    const int NOPROCESSIMAGE=2;

    if (mode==PROCESSIMAGE)
      cv::resize(in, out, Size(in.cols/4*3, in.rows/4*3), 0, 0, INTER_LINEAR);
    else if (mode==NOPROCESSIMAGE)
      out=in;
    else 
      {
	cout << "no valid image processing mode is available" << endl;
	out=in;
      }
  }
  
  // Function: to find the maximum face in given image
  // in: input image
  // out: processed image
  // face: location and size of the face
  // faceFrame: face with interested region
  void findFace(Mat in, Mat & out, Rect & face, Mat & faceFrame, int& face_x, int& face_y, int& face_height, int& face_width)
  {
    vector<Rect_<int> > faces;
    haar_faces(in, faces);

    if (faces.size()>0)
      {
	size_t n;
	findMaxFace(faces, n);

	Rect face=faces[n];

	findProcessRegion(in, face, faceFrame);

	face_x = face.x;
	face_y = face.y;
	face_height = face.height;
	face_width = face.width;

	//drawFace(in, face, "face");
	out = in;
      }
    else
      {
	faceFrame=in(Rect(face.x, face.y, face.width, face.height));
	out = in;
      }
  }

  // Function: process tracked face to desire face
  // in: input image
  // face: location and size of the face
  // out: processed face
  void findProcessRegion(Mat in, Rect & face, Mat & out)
  {
    // Main Face Region
    face.x=face.x+face.width/4;
    face.width=face.width/2;

    face.height=(int)(face.height*0.80);

    Mat faceFrame=in(face);

    // Remove eye region
    
    Rect up(0,0,face.width,face.height);
    Rect down=up;

    up.height=(int)(up.height*0.20);
    out=faceFrame(up);

    down.y=down.y+(int)(down.height*0.55);
    down.height=(int)(down.height*0.45);
    Mat downFrame=faceFrame(down);

    out.push_back(downFrame);
    cv::cvtColor(out, out, CV_BGR2GRAY);
  }

  // Function: draw tracked face
  // in: input frame
  // face_n: location and size of the face
  // box_text: title of the face
  void drawFace(Mat & in, cv::Rect & face_n, string box_text)
  {
    rectangle(in, face_n, CV_RGB(0,255,0), 1);
    int pos_x=std::max(face_n.tl().x-10, 0);
    int pos_y=std::max(face_n.tl().y-10, 0);
    putText(in, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0),2);
  }

  // Function: find the maximum face from all acquired faces
  // faces: all acquired faces' information
  // n: maximum face
  void findMaxFace(vector<Rect_<int> > faces, size_t & n)
  {
    n=0;
    int max=-1;
    for (size_t i=0; i<faces.size(); i++)
      {
	if (max<faces[i].height*faces[i].width)
	  {
	    max=faces[i].height*faces[i].width;
	    n=i;
	  }
      }
  }

  // Function: track face
  // in: input image
  // faces: all tracked faces' information
  void haar_faces(Mat in, vector<cv::Rect_<int> > & faces)
  {
    Mat gray;
    cv::cvtColor(in, gray, CV_BGR2GRAY);

    haar_cascade.detectMultiScale(gray, faces, 1.1, 2);
    gray.release();
  }

  // Function: perform LK Optical Flow algorithm
  // im1: previous image
  // im2: next image
  // cornerA: feature points position in prev frame
  // cornerB: feature points position in next frame
  // win_size: window size
  // MAX_CORNERS: maximum tracked feature points
  void performLKFilter(Mat im1, Mat im2, vector<Point2f> cornerA, vector<Point2f> & cornerB, int win_size/*, const int MAX_CORNERS*/)
  {
	/// Apply Histogram Equalization
	if(!im1.empty())
	{
		cvtColor( im1, im1, CV_BGR2GRAY );
		equalizeHist( im1, im1 );
	}
	if(!im2.empty())
	{
		cvtColor( im2, im2, CV_BGR2GRAY );
		equalizeHist( im2, im2 );
	}

    Size img_sz=im1.size();

    std::vector<uchar> feature_found; 
    feature_found.reserve(cornerA.size() /*MAX_CORNERS*/);
    std::vector<float> feature_errors; 
    feature_errors.reserve(cornerA.size() /*MAX_CORNERS*/);
	calcOpticalFlowPyrLK(im1, im2, cornerA, cornerB, feature_found, feature_errors, Size(win_size, win_size), 5, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.001));
  }


  // Function: interpolation 1D data using cubic interpolation
  // in: input matrix in 1*N
  // out: output interpolated matrx in 1*M (M=round(N*outFr/inFr))
  // inFr: original sampling frequency
  // outFr: desired sampling frequency
  void interpolationOneRow(Mat in, Mat & out, double inFr, double outFr)
  {
    double Fr=outFr/inFr;

	int method = INTER_CUBIC; //method for upsampling
	if(Fr<1){method = INTER_AREA;} //method for downsampling

    cv::resize(in, out, Size((int)((double)in.cols*Fr), in.rows), 0, 0, method);
  }

  // Function: apply interpolation for all data
  // in: original data matrix (M*N)
  // out: interpolated output matrix (M*(N*outFr/inFr))
  // printmsg: to check if interpolation is working.
  void interpolation(Mat in, Mat & out, double inFr, double outFr, bool printmsg)
  {
    int inrows=in.rows;

    for (int i=0;i<inrows;i++)
      {
	Mat in_row=in.row(i);
	Mat out_row;
	interpolationOneRow(in_row, out_row, inFr, outFr);
	out.push_back(out_row);
      }

    // for checking only
    if (printmsg)
      {
	cout << "[INPUT] Rows: " << in.rows << " Cols: " << in.cols << endl;
	cout << "[OUTPUT] Rows: " << out.rows << " Cols: " << out.cols << endl;
      }
  }

  //floor all the values in a matrix
  void floorMatrix(Mat& in) 
  {
	  for(int i=0;i<in.rows;i++)
	  {
		  for(int j=0;j<in.cols;j++)
		  {
			  in.at<float>(i,j) = abs(floor(in.at<float>(i,j)));
		  }
	  }
  }

  //Calculate the mode of values in an array
  float GetMode(float daArray[], int iSize) {
    // Allocate an int array of the same size to hold the
    // repetition count
    int* ipRepetition = new int[iSize];
    for (int i = 0; i < iSize; ++i) {
        ipRepetition[i] = 0;
        int j = 0;
        //bool bFound = false;
        while ((j < i) && (daArray[i] != daArray[j])) {
            if (daArray[i] != daArray[j]) {
                ++j;
            }
        }
        ++(ipRepetition[j]);
    }
    int iMaxRepeat = 0;
    for (int i = 1; i < iSize; ++i) {
        if (ipRepetition[i] > ipRepetition[iMaxRepeat]) {
            iMaxRepeat = i;
        }
    }
    delete [] ipRepetition;
    return daArray[iMaxRepeat];
}

  // Function: discard signal from the matrix which are unstable: maximum inter-frame distance is bigger than mode of inter-frame distances for a tracked point
  void discardErraticSignals(Mat in, Mat & out, bool printMsg)
  {
	  Mat stepSize = in(cv::Rect(1,0,in.cols-1,in.rows)) - in(cv::Rect(0,0,in.cols-1,in.rows));
	  Mat fStepSize = stepSize;
	  floorMatrix(fStepSize); //Round differences down to whole pixel -> to calculate mode of difference-series

	  for(int i=0;i<fStepSize.rows;i++)
	  {
		  Mat hist;
		  int maxVal;
		  double dmaxVal, dminVal;

		  //cout << "Signal " << i << ":" << endl;
		  minMaxLoc(fStepSize.row(i), &dminVal, &dmaxVal);
		  //cout <<"Max: " << dmaxVal << endl; //Calculating differences and maximum diff works so far!
		  maxVal = (int)dmaxVal;

		  float* daArray;
		  daArray = new float [fStepSize.cols];
		  for (int j=0;j<fStepSize.cols;j++)
		  {
			  daArray[j] = fStepSize.at<float>(i,j); //convert matrix to array
		  }
		  int mode = (int)GetMode(daArray, fStepSize.cols);
		  delete[] daArray;

		  if (printMsg)
		  {
			  std::cout << "Signal " << i << ": Mode " << mode << ", MaxVal " << maxVal << endl;
		  }

		  if (maxVal <=  mode)
		  {
			  out.push_back(in.row(i));
		  }
	  }
	  
	  if(out.rows < 2){
		in.copyTo(out);
		cout << "No stable points available. Using everything.." << endl;
	  }
	  else if(printMsg)
	  {
		cout << "Stable signals: " << out.rows << endl;
	  }
	  
  }

  // FROM: http://mechatronics.ece.usu.edu/yqchen/filter.c/FILTER.C
  // Function: C implemenation of Matlab filter
  // ord: order (ord=order*2)
  // a: butter parameter a
  // b: butter parameter b
  // np: size of signal
  // x: input signal
  // y: filtered signal
  void filter(int ord, double *a, double *b, int np, double *x, double *y)
  {
    y[0]=b[0]*x[0];
    for (int i=1;i<ord+1;i++) 
      {
        y[i]=0.0;
        for (int j=0;j<i+1;j++)
	  y[i]=y[i]+b[j]*x[i-j];
        for (int j=0;j<i;j++)
	  y[i]=y[i]-a[j+1]*y[i-j-1];
      }

    for (int i=ord+1;i<np;i++)
      {
	y[i]=0.0;
        for (int j=0;j<ord+1;j++)
	  y[i]=y[i]+b[j]*x[i-j];
        for (int j=0;j<ord;j++)
	  y[i]=y[i]-a[j+1]*y[i-j-1];
      }
  }

  //got code from internet for butterworth filter coeficients, generalized it a bit
 double *ComputeLP( int FilterOrder )
{
    double *NumCoeffs;
    int m;
    int i;

    NumCoeffs = (double *)calloc( FilterOrder+1, sizeof(double) );
    if( NumCoeffs == NULL ) return( NULL );

    NumCoeffs[0] = 1;
    NumCoeffs[1] = FilterOrder;
    m = FilterOrder/2;
    for( i=2; i <= m; ++i)
    {
        NumCoeffs[i] =(double) (FilterOrder-i+1)*NumCoeffs[i-1]/i;
        NumCoeffs[FilterOrder-i]= NumCoeffs[i];
    }
    NumCoeffs[FilterOrder-1] = FilterOrder;
    NumCoeffs[FilterOrder] = 1;

    return NumCoeffs;
}
 double *ComputeHP( int FilterOrder )
{
    double *NumCoeffs;
    int i;

    NumCoeffs = ComputeLP(FilterOrder);
    if(NumCoeffs == NULL ) return( NULL );

    for( i = 0; i <= FilterOrder; ++i)
        if( i % 2 ) NumCoeffs[i] = -NumCoeffs[i];

    return NumCoeffs;
}
 double *ComputeNumCoeffs(int FilterOrder,double Lcutoff, double Ucutoff, double *DenC) 
{
    double *TCoeffs;
    double *NumCoeffs;
    std::complex<double> *NormalizedKernel;
    //double Numbers[2*FilterOrder+1]={0,1,2,3,4,5,6,7,8,9,10};
	
	//double* Numbers = new (nothrow) double [2*FilterOrder+1];
	//for(int i=0;i<2*FilterOrder+1;i++)
	//{
	//	Numbers[i] = i;
	//}

    int i;

    NumCoeffs = (double *)calloc( 2*FilterOrder+1, sizeof(double) );
    if( NumCoeffs == NULL ) return( NULL );

    NormalizedKernel = (std::complex<double> *)calloc( 2*FilterOrder+1, sizeof(std::complex<double>) );
    if( NormalizedKernel == NULL ) return( NULL );

    TCoeffs = ComputeHP(FilterOrder);
    if( TCoeffs == NULL ) return( NULL );

    for( i = 0; i < FilterOrder; ++i)
    {
        NumCoeffs[2*i] = TCoeffs[i];
        NumCoeffs[2*i+1] = 0.0;
    }
    NumCoeffs[2*FilterOrder] = TCoeffs[FilterOrder];
    double cp[2];
    double Bw, Wn;
    cp[0] = 2*2.0*tan(PI * Lcutoff/ 2.0);
    cp[1] = 2*2.0*tan(PI * Ucutoff / 2.0);

    Bw = cp[1] - cp[0];
    //center frequency
    Wn = sqrt(cp[0]*cp[1]);
    Wn = 2*atan2(Wn,4);
    //double kern;
    const std::complex<double> result = std::complex<double>(-1,0);

    for(int k = 0; k<2*FilterOrder+1; k++)
    {
        NormalizedKernel[k] = std::exp(-sqrt(result)*Wn*(double)k);
    }
    double b=0;
    double den=0;
    for(int d = 0; d<2*FilterOrder+1; d++)
    {
        b+=real(NormalizedKernel[d]*NumCoeffs[d]);
        den+=real(NormalizedKernel[d]*DenC[d]);
    }
    for(int c = 0; c<2*FilterOrder+1; c++)
    {
        NumCoeffs[c]=(NumCoeffs[c]*den)/b;
    }

    free(TCoeffs);
    return NumCoeffs;
}
 double *ComputeDenCoeffs( int FilterOrder, double Lcutoff, double Ucutoff )
{
    int k;            // loop variables
    double theta;     // PI * (Ucutoff - Lcutoff) / 2.0
    double cp;        // cosine of phi
    double st;        // sine of theta
    double ct;        // cosine of theta
    double s2t;       // sine of 2*theta
    double c2t;       // cosine 0f 2*theta
    double *RCoeffs;     // z^-2 coefficients
    double *TCoeffs;     // z^-1 coefficients
    double *DenomCoeffs;     // dk coefficients
    double PoleAngle;      // pole angle
    double SinPoleAngle;     // sine of pole angle
    double CosPoleAngle;     // cosine of pole angle
    double a;         // workspace variables

    cp = cos(PI * (Ucutoff + Lcutoff) / 2.0);
    theta = PI * (Ucutoff - Lcutoff) / 2.0;
    st = sin(theta);
    ct = cos(theta);
    s2t = 2.0*st*ct;        // sine of 2*theta
    c2t = 2.0*ct*ct - 1.0;  // cosine of 2*theta

    RCoeffs = (double *)calloc( 2 * FilterOrder, sizeof(double) );
    TCoeffs = (double *)calloc( 2 * FilterOrder, sizeof(double) );

    for( k = 0; k < FilterOrder; ++k )
    {
        PoleAngle = PI * (double)(2*k+1)/(double)(2*FilterOrder);
        SinPoleAngle = sin(PoleAngle);
        CosPoleAngle = cos(PoleAngle);
        a = 1.0 + s2t*SinPoleAngle;
        RCoeffs[2*k] = c2t/a;
        RCoeffs[2*k+1] = s2t*CosPoleAngle/a;
        TCoeffs[2*k] = -2.0*cp*(ct+st*SinPoleAngle)/a;
        TCoeffs[2*k+1] = -2.0*cp*st*CosPoleAngle/a;
    }

    DenomCoeffs = TrinomialMultiply(FilterOrder, TCoeffs, RCoeffs );
    free(TCoeffs);
    free(RCoeffs);

    DenomCoeffs[1] = DenomCoeffs[0];
    DenomCoeffs[0] = 1.0;
    for( k = 3; k <= 2*FilterOrder; ++k )
        DenomCoeffs[k] = DenomCoeffs[2*k-2];


    return DenomCoeffs;
}
 double *TrinomialMultiply( int FilterOrder, double *b, double *c )
{
    int i, j;
    double *RetVal;

    RetVal = (double *)calloc( 4 * FilterOrder, sizeof(double) );
    if( RetVal == NULL ) return( NULL );

    RetVal[2] = c[0];
    RetVal[3] = c[1];
    RetVal[0] = b[0];
    RetVal[1] = b[1];

    for( i = 1; i < FilterOrder; ++i )
    {
        RetVal[2*(2*i+1)]   += c[2*i] * RetVal[2*(2*i-1)]   - c[2*i+1] * RetVal[2*(2*i-1)+1];
        RetVal[2*(2*i+1)+1] += c[2*i] * RetVal[2*(2*i-1)+1] + c[2*i+1] * RetVal[2*(2*i-1)];

        for( j = 2*i; j > 1; --j )
        {
            RetVal[2*j]   += b[2*i] * RetVal[2*(j-1)]   - b[2*i+1] * RetVal[2*(j-1)+1] +
                c[2*i] * RetVal[2*(j-2)]   - c[2*i+1] * RetVal[2*(j-2)+1];
            RetVal[2*j+1] += b[2*i] * RetVal[2*(j-1)+1] + b[2*i+1] * RetVal[2*(j-1)] +
                c[2*i] * RetVal[2*(j-2)+1] + c[2*i+1] * RetVal[2*(j-2)];
        }

        RetVal[2] += b[2*i] * RetVal[0] - b[2*i+1] * RetVal[1] + c[2*i];
        RetVal[3] += b[2*i] * RetVal[1] + b[2*i+1] * RetVal[0] + c[2*i+1];
        RetVal[0] += b[2*i];
        RetVal[1] += b[2*i+1];
    }

    return RetVal;
}
  /////END code from internet for butterworth

 //Function: substract the mean of the signal from each element, this is apparently required for filtering
  void meanSignal(Mat in, Mat& out)
  {
	cv::Scalar signalMean = 0;
	float meanValue = 0;

	for(int z=0; z<in.rows; z++){ //print all processed signals
		signalMean=cv::mean(in.row(z));
		meanValue=(float)signalMean[0];
		out.row(z) = in.row(z) - meanValue;
    }

  }

  //outputs a CSV file to the specified folder with the specified name
  //input must be a matrix with float elements
  void exportSig(Mat signal, string folder, string filename)
  {
	ofstream exportFile(folder + filename + ".csv");
		for(int z=0; z<signal.rows; z++){ //print all processed signals
			for(int w = 0; w<signal.cols; w++)
		    {
  				exportFile << signal.at<float>(z,w) << ", ";
  		    }
		    exportFile << /*exportFile << signal.at<float>(z,signal.cols-1) << */endl;
		}
	exportFile.close();
  }

  //apply 5th order butter worth filter with passband of [0.75, 5]Hz to each row of the input matrix
  void applyFilter(Mat in, Mat & out, int FrameRate, bool printMsg)
  {
	  double Fr = (double) FrameRate;
      
	 
	  int FilterOrder = 5;
	  double Lcutoff = (double)2*0.75/Fr;//we have 250 or 61 Hz sampling so divide these by it //0.0246
	  double Ucutoff = (double)2*5/Fr;

	  if(printMsg)
	  {
	  cout << "Fr = " << Fr << endl;
	  cout << "LeftCutOff = " << Lcutoff << endl;
	  cout << "RightCutOff = " << Ucutoff << endl;
	  }

	  /////////////////////////COMPUTE COEFFS
	  double* DenC = ComputeDenCoeffs(FilterOrder, Lcutoff, Ucutoff);
	  double* NumC = ComputeNumCoeffs(FilterOrder, Lcutoff, Ucutoff, DenC);
	  //double scaleFactor =  sf_bwbp(FilterOrder, Lcutoff, Ucutoff); //calc scaling factor
	  ////////////////////////END COMPUTE COEFFS

	  const int sigLength = in.cols;
	  double* input = new (nothrow) double [sigLength]; //reserve space in mem
	  double* output = new (nothrow) double [sigLength];

	  for(int i=0;i<in.rows;i++){
		
		for(int j=0;j<sigLength;j++){ //convert matrix row to array of doubles..
			input[j] = (double)(in.at<float>(i,j));
		}

		filter(2*FilterOrder, DenC, NumC, sigLength, input, output); //filter signal with butterworth

		for(int j=0;j<sigLength;j++){ //convert matrix row to array of doubles..
			out.at<float>(i,j) = (float)(output[j]);
		}
		
	  }
	  delete[] input; delete[] output; //free memory
  }

  // in: input filtered matrix (M*N)
  // out: PCA projection (data in rows)
  void calculatePCAProjection(Mat in, Mat & out, float alpha, bool printMsg)
  {
	//FIRST THROW OUT FIRST ALPHA = 25% OF TIME-POINTS WITH BIGGEST L2-NORM 
	Mat Norms = Mat::zeros(1,in.cols, CV_32F); //norm of the vector for each time-frame
	Mat selected = Mat::ones(1, in.cols, CV_32F); //boolean whether a time-frame is selected
	Mat temp = Mat::zeros(0,in.rows, CV_32F); //matrix with alpha% of time-frames tossed out	
	for(int i=0; i<in.cols;i++)
	{
		Norms.at<float>(0,i) = (float)norm(in.col(i), NORM_L2);
	}

	int outputCols = (int)((1-alpha)*(float)in.cols);
	int maxIdx[2];

	for(int i=0;i<(in.cols - outputCols);i++)
	{
		cv::minMaxIdx(Norms, 0, 0, 0, maxIdx);
		selected.at<float>(0,maxIdx[1]) = 0; //throw out max-norm-column
		Norms.at<float>(0,maxIdx[1]) = 0; //and set the norm entry to 0...
	}

	Mat entry; //temp mat
	for(int i=0; i<in.cols;i++) //make new matrix with desired time-frames
	{
		if((int)selected.at<float>(0,i) == 1)
		{ 
			entry = in.col(i).t();
			temp.push_back(entry);
		}
	}
	temp = temp.t();

	//and now get the PCA-eigenvectors. after this take entire signal again for reconstruction in terms of these eigenvectors:
	int maxComponents = 15; //15
    PCA pca(temp, Mat(), CV_PCA_DATA_AS_COL, maxComponents);
	pca.project(in, out); 

	if(printMsg)
	{
		cout << "Eigenvalues: " << pca.eigenvalues(Rect(0,0,1,10)) << endl;
	}
  }

  // Function: select best quality signal
  // Best periodic signal -> highest percentage of total spectral power accounted for by the frequency with maximal power and its first harmonic
  // Only look at first n_sig signals
  int findBestSignalFourier(Mat in, double Fr, int n_sig, float & heartRate, string outputFolder, bool printMsg)
  {
	  ofstream csvoutput(outputFolder + "FFTsignal.csv");
	  Mat signalQuality = Mat::zeros( 1, n_sig, CV_32F);
	  Mat maxFreq = Mat::zeros( 1, n_sig, CV_32F);

	  for (int i=0;i<n_sig;i++) //Only look at first n_sig signals
	  {
		    Mat signal = in.row(i);
		    cv::Scalar signalMean=cv::mean(signal);
			double meanValue=signalMean[0];
			signal=signal-meanValue; //first substract mean of signal

			// fft
			Mat complexI;
			Mat planes[] = {Mat_<float>(signal), Mat::zeros(signal.size(), CV_32F)};
			merge(planes, 2, complexI);
			cv::dft(complexI, complexI, DFT_COMPLEX_OUTPUT);
    
			split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
			magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
			Mat magI = planes[0];

			Mat halfSpec = magI(cv::Rect(0,0,(int)floor(magI.cols/2)+1,1));

			Mat PX;
			cv::pow(halfSpec,2,PX);
			normalize(PX, PX, 1, 0, NORM_L1);

			for (int j=0; j<halfSpec.cols; j++){
					csvoutput << PX.at<float>(0,j) << ", ";
			}
			csvoutput << "\n";

			double maxPower;
			int maxIdx[2];
			cv::minMaxIdx(PX, 0, &maxPower, 0, maxIdx);
			signalQuality.at<float>(0,i) = PX.at<float>(0,maxIdx[1]) + PX.at<float>(0,2*maxIdx[1]); //signal quality = power in max freq + first harmonic of max freq / signal power
			maxFreq.at<float>(0,i) = (float)(((double) maxIdx[1] )*Fr/((double)in.cols));
	  }

		for (int i=0; i<=(int)floor(in.cols/2); i++){ //print frequencies to plot on x-axis to CSV-file
				csvoutput << ((double)i)*Fr/((double)in.cols) << ", ";
		}
		csvoutput.close();

	  //pick best signal to return
	  int bestSignal[2];
	  cv::minMaxIdx(signalQuality, 0, 0, 0, bestSignal);

	  if(printMsg){
	  for (int j = 0; j<n_sig; j++) {
		  cout << "Quality " << j << ": " << signalQuality.at<float>(0, j) << endl;
	  }
	  for (int j = 0; j < n_sig; j++) {
		  cout << "MaxFreq " << j << ": " << maxFreq.at<float>(0,j)*60 << " BPM"  << endl;
	  }
	  }
	  heartRate = (float)maxFreq.at<float>(0,bestSignal[1])*60;
	  cout << "Estimated FFT heartrate: " << heartRate << "BPM" << endl;

	  return bestSignal[1];
  }

  size_t getOptimalDCTSize(size_t N) { return 2*getOptimalDFTSize((N+1)/2); }

  // Function: select best quality signal
  // Best periodic signal -> weird method described in "Improved pulse detection from head motions using DCT" by R. Irani (http://vbn.aau.dk/ws/files/124549539/ri.pdf) 
  int findBestSignalDCT(Mat in, double Fr, int n_sig, float & heartRate, string outputFolder, bool printMsg)
  {

	int optDCTsize = getOptimalDCTSize(in.cols);

	Mat dct_sig = Mat::zeros(in.rows, optDCTsize, CV_32F);
	Mat dct_in = Mat::zeros(in.rows, optDCTsize, CV_32F);
	in.copyTo(dct_in(cv::Rect(0, 0, in.cols, in.rows)));

	dct(dct_in, dct_sig, DCT_ROWS); //DCT transform all signals

	this->exportSig(dct_sig, outputFolder, "DCTsignals");

	//Now select best signal..
	Mat sig_total_energy = Mat::zeros(1, in.rows, CV_32F); //first calc total energy of each signal
	Mat qualitySignal = Mat::zeros(1, in.rows, CV_32F);
	Mat minKsignal = Mat::zeros(1, in.rows, CV_32F);
	Mat dct_squared;
	pow(dct_sig, 2, dct_squared); //square signal to get power
	
	for (int i=0;i<in.rows;i++) //for every signal S_i
	{
		Mat maxPowerCoeffs = Mat::Mat(0,1,CV_32F);
		sig_total_energy.at<float>(0,i) = (float)norm(dct_squared.row(i), NORM_L1); //total energy of this row

		float partPower = 0; //part of the power with the components so far
		while(partPower < 0.5 * sig_total_energy.at<float>(0,i)) //get DCT components until 50% of the signals power is accounted for
		{ 
			double maxPower;
			int maxIdx[2];
			cv::minMaxIdx(dct_squared.row(i), 0, &maxPower, 0, maxIdx);

			maxPowerCoeffs.push_back((float)maxIdx[1]);
			partPower += (float)maxPower;//add component to part of power currently accounted for and set the component to 0 (to find next biggest signal next time)
			dct_squared.at<float>(i,maxIdx[1]) = 0;
		}

		//pick the 5 maxPowerCoeffs with smallest index (such that 2* the coefficient can be found in the DCT signal, so coeff should be smaller than 2* the DCT length?):
		Mat K = Mat::Mat(0,1,CV_32F);
		Mat DCTvalues = Mat::Mat(0,1,CV_32F); //vector of DCT entries for selected indices and first harmonic. used to calc quality.
		Mat curPowerCoeffs;
		maxPowerCoeffs.copyTo(curPowerCoeffs); //copy this dct signals row so we can change it without further consequences
		for(int j=0; K.rows < 5 && j < maxPowerCoeffs.rows; j++)
		{
			double power;
			int minIdx[2];
			cv::minMaxIdx(curPowerCoeffs, &power, 0, minIdx, 0);
			
			if(minIdx[0] < (int)floor(in.cols/2))
			{
				K.push_back((float)power);
				DCTvalues.push_back(dct_sig.at<float>(i,minIdx[0]));
				DCTvalues.push_back(dct_sig.at<float>(i,2*minIdx[0]));
				curPowerCoeffs.at<float>(minIdx[0]) = 99999; //to ignore it in further iterations

				//if(j==0) //save smallest Kh for calculation heartbeat
				//{
				//	minKsignal.at<float>(1,i) = minIdx[0];
				//}

			}
		}

		qualitySignal.at<float>(0,i) = (float)norm(DCTvalues, NORM_L2)/(float)norm(dct_sig.row(i), NORM_L2);
		minKsignal.at<float>(0,i) = K.at<float>(0,0);

		if(printMsg)
		{
			cout << "maxPowerCoeffs: " << maxPowerCoeffs << endl;
			cout << "K: " << K << endl << "min(K): " << minKsignal.at<float>(0,i) << endl;
			cout << "DCTvalues: " << DCTvalues << endl;
			system("pause");
		}
	}

	if(printMsg){
		cout << "Signal quality: " << qualitySignal << endl;
		cout << "Min K signal: " << minKsignal << endl;
	}

	//Find highest quality:
	double maxQuality;
	int maxIdx[2];
	cv::minMaxIdx(qualitySignal, 0, &maxQuality, 0, maxIdx);
	int bestSignal = maxIdx[1];

	//Take min of the set of K indices, and place this in a signal of otherwise zeros:
	Mat sigToUse = Mat::zeros(1,optDCTsize, CV_32F);
	Mat beatSignal;
	int coeffLoc = (int)minKsignal.at<float>(0,bestSignal);
	sigToUse.at<float>(0, coeffLoc) = dct_sig.at<float>(bestSignal, coeffLoc);
	idct(sigToUse, beatSignal, DCT_ROWS); //inverse discrete cosine transform to get a pure sine wave
	//this->exportSig(beatSignal, outputFolder, "DCTfilteredSig");

	//Fourier to find max freq of this wave:
	Mat complexI;
	Mat planes[] = {Mat_<float>(beatSignal), Mat::zeros(beatSignal.size(), CV_32F)};
	merge(planes, 2, complexI);
	cv::dft(complexI, complexI, DFT_COMPLEX_OUTPUT);
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];
	Mat halfSpec = magI(cv::Rect(0,0,(int)floor(magI.cols/2)+1,1));
	cv::minMaxIdx(halfSpec, 0, 0, 0, maxIdx); //get max freq from fourier spectrum

	//this->exportSig(halfSpec, outputFolder, "FFTDCTfilteredSig");

	heartRate = (float)(((double) maxIdx[1] )*Fr/((double)beatSignal.cols)) * 60; //this max freq is the heartrate freq
	cout << "Estimated DCT heart rate: " << heartRate << "BPM." << endl;

	return bestSignal;
  }

  //Function: cut signals into small pieces of timelength T, and export these to CSV to train a self-organising map in MATLAB
  void exportTrainingData(Mat signal, double Fr, double T, string folder, string filename)
  {
	ofstream exportFile;
	exportFile.open(folder + filename + ".csv", fstream::out | fstream::app);
	float initVal = 0; //mean to substract from startpoint of every signal-segment, to make all segments start at 0

		for(int z=0; z<signal.rows; z++){ //print all processed signals

			initVal = signal.at<float>(z,0);
			for(int w = 1; w<signal.cols; w++)
		    {

  				exportFile << signal.at<float>(z,w-1) - initVal;
				if(w % (int)(T*Fr) == 0){ 
					if(signal.cols-w < (int)(T*Fr)){break;}
					exportFile << endl; //print a newline as well
					initVal = signal.at<float>(z,w); //set startingpoint of segment to 0
				} 
				else { exportFile << ", "; }
  		    }
		    exportFile << /*signal.at<float>(z,(signal.cols-1)) <<*/ endl;
		}
	exportFile.close();
  }

  //Function: use a windowsize to mark peaks in a signal by marking at as a peak if its the highest point in the window
  void peakDetect(Mat in, Mat& peaks, double Fr, int windowSize)
  {
	  for (int i=0;i<in.cols ; i++){ //check for every window the biggest signal

		  int sigStart = max(0,(int)(i-ceil(windowSize/2))); //determine signal window
		  int sigEnd = min(in.cols-1, (int)(i+ceil(windowSize/2)));
		  Rect window = cv::Rect(sigStart,0, sigEnd-sigStart,1);
		  Mat windowedSignal = in(window);

		  double dmaxVal, dminVal;
		  minMaxLoc(windowedSignal, &dminVal, &dmaxVal); //determine if i is local max within window. if so, add to ECG-peaks

		  if(in.at<float>(0,i) >= dmaxVal){
			  peaks.push_back((float)i/(float)Fr);
		  }
	  }
  }

  // Function: calculate heart beat in minute
  // in: Input signal
  // heart_beat: output heart beat (Hz)
  // Fr: sampling rate
  void calculateHeartRate(Mat in, double & heart_beat, double Fr, float heartRate)
  {
	//Peak detection
	Mat peaks;

	heartRate = max((float)1, heartRate); //To prevent division by 0
	int windowSize = (int)max((double)3, floor((Fr/(double)heartRate)*60)); //45 is min pulse rate?
    peakDetect(in, peaks, Fr, windowSize);

	heart_beat = ((double)peaks.rows/ (double)in.cols) * Fr * (double)60;
	cout << "On basis of peaks the estimated heart rate is: " << heart_beat << "BPM." << endl;
  }

  //-------------------PAPER Balakrishnan:
  //point tracking
  //select vertical component of this time-series
  //upsample to 250Hz (Do this after the filtering, else the filter gets unstable)
  //find maximum distance between tracked points between frames and discard points which distance exceeds the mode of the distribution
  //apply 5th order Butterworth filter with passband of [0.75, 2]Hz
  //PCA decomposition
  //signal selection: Best periodic signal -> highest percentage of total spectral power accounted for by the frequency with maximal power and its first harmonic
  //peak detection: point is a peak if its the maximum in a window centered around it, with window size round(f_sample/f_pulse), where f_sample = 250Hz
};
