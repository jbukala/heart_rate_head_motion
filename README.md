# pulse-head-motions
## Determine heart rate from head motions using a video. 
#### Requirements: Needs OpenCV installed and IntraFace v1.2 present. 
MATLAB-scripts used to run algorithm on the MAHNOB-HCI database. 
OpenCV's GFT function and IntraFace's landmark points used to track points.
Uses time-series of these tracked points with a filtering and clustering approach to find the frequency at which the head moves due to the arterial pressure.
