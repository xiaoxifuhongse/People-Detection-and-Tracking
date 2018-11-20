#include "CFollowerRobot.h"
#define  LIVE 1
CFollowerRobot::CFollowerRobot()
{
	
	// parameters single candidates tracking
	tracker.legSigmaZ = 0.02; // position measure uncertainty(m)
	tracker.legSigmaP = 0.2;  // leg probability measure uncertainty
	tracker.legSigmaAcc = 6;  // model uncertainty, taking into account acceleration (m/s^2)

	// parameters people tracking
	tracker.peopleSigmaZ = 0.05;
	tracker.peopleSigmaP = 0.2;
	tracker.peopleSigmaAcc = 6;
	tracker.peopleDistThreshold = 100;//75;
	tracker.legProbabilityThreshold = 0.8;
	
	tracker.currentTimestamp = (double)getTickCount();
	tracker.oldTimestamp     = (double)getTickCount();
	// parameter to sepcify how often searching for a floor plane
	tracker.refreshIntervalFloorPlane = 0.05; // seconds
	
	// floor plane tolerance in mm
	tracker.floorPlaneTolerance = 2;

	tracker.floorPlaneTimestamp = -FLT_MAX;
	tracker.legFreeID = 0;
	tracker.peopleFreeID = 0;
	tracker.pose.x = 0;
	tracker.pose.y = 0;
	tracker.pose.z = 0;	
}

void CFollowerRobot::setupTracker()
{
#if LIVE
	xn::EnumerationErrors errors;
	XnStatus nRetVal = XN_STATUS_OK;
	nRetVal = g_Context.InitFromXmlFile("SensorConfig.xml", &errors);
	if(nRetVal == XN_STATUS_NO_NODE_PRESENT)
	{
		XnChar strError[1024];
		errors.ToString(strError,1024);
		printf("%s\n",strError);
	}
	else if(nRetVal != XN_STATUS_OK)
	{
		printf("Open failed: %s\n",xnGetStatusString(nRetVal));
	}

	else
	{
		printf("open ni init surcessed!\n");
	}
	// Detect Image
	nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_IMAGE, g_Image);
	if (nRetVal != XN_STATUS_OK)								
	{	
		printf("Image Node : Not found\n");
	}
	else
	{
		printf("Image Node : Found\n");

	}
	// Detect Depth
	nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_DepthGenerator);
	if (nRetVal != XN_STATUS_OK)								
	{	
		printf("Depth Node : Not found\n");
	}
	else
	{
		printf("Depth Node : Found\n");

	}

#else
	XnStatus nRetVal = XN_STATUS_OK;
	
	nRetVal=g_Context.Init();  

	nRetVal = g_Context.OpenFileRecording("S-Difficult.oni");//  S-Medium.oni    S-Difficult.oni
	  
	

	nRetVal=g_Context.FindExistingNode(XN_NODE_TYPE_IMAGE,g_Image);     
	if (nRetVal != XN_STATUS_OK)								
	{	
		printf("Image Node : Not found\n");
	}
	else
	{
		printf("Image Node : Found\n");

	}  

	nRetVal=g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH,g_DepthGenerator);
	if (nRetVal != XN_STATUS_OK)								
	{	
		printf("Depth Node : Not found\n");
	}
	else
	{
		printf("Depth Node : Found\n");

	}
#endif
	g_DepthGenerator.GetMetaData(depthMD);
	width  = depthMD.FullXRes();
    height = depthMD.FullYRes();

	rgb = Mat::zeros(Size(width, height), CV_8UC3);
	depth = Mat::zeros(Size(width, height), CV_16UC1);

}

void CFollowerRobot::update()
{
	g_Context.WaitAndUpdateAll();
	g_DepthGenerator.GetMetaData(depthMD);
	g_Image.GetMetaData(imageMD);
	memcpy(rgb.data, imageMD.Data(),width*height*3);
	memcpy(depth.data, depthMD.Data(), width*height*2);
	cvtColor(rgb, rgb, CV_RGB2BGR);
	tracker.oldTimestamp = tracker.currentTimestamp;
	tracker.currentTimestamp = (double)getTickCount();

}
void CFollowerRobot::trackPeople()
{
	
	update();
	// point cloud downsampling using voxel grid filter
	cPointCloud.getVoxelGrid(depth, rgb);

	//search for floor plane if refreshIntervalFloorPlane has elapsed since
    // the last floor update

	//PlaneParmeters updatePlane;
	//cout<<(tracker.currentTimestamp - tracker.floorPlaneTimestamp)/getTickFrequency()<<endl;
	if(((tracker.currentTimestamp - tracker.floorPlaneTimestamp)/getTickFrequency())>0.5)
	{
		if(!cPointCloud.getGroundPlane(tracker.floorPlane, 20, tracker.floorPlaneTolerance))
			return;
		tracker.floorPlaneTimestamp = tracker.currentTimestamp;
	}

	// transform the point cloud to a more pratical referenct system
	Mat Rinv;
	cPointCloud.rotatePointCloud(tracker.floorPlane, Rinv);

	// search candidates legs in the point cloud
	vector<Candidate>candidates;
	cPointCloud.getCandidateLegs(candidates);
	
	//extract features and predict candidates leg probabilities
	Mat probEstimates;
	cPointCloud.predictClass(candidates, probEstimates);
	
	
	vector<int>candidatesTracks(candidates.size(),-1);
	cTracks.candidatesTrackerPDAF(candidates,probEstimates,tracker,candidatesTracks);
	 
	cTracks.peopleTrackerPDAFOnLegTracks(candidates,candidatesTracks,tracker);

	////// field of view on the map
	float yMax = 3.5;
	float fieldOfView =(float)57/180*CV_PI;
	float xMin = -tan(fieldOfView/2)*yMax;
	float xMax = tan(fieldOfView/2)*yMax;
	float yMin = 0;

	Mat img = Mat::zeros(Size(640,480),CV_8UC3);
	float xOriMin = int((-tan(fieldOfView/2)*0.5)*320/2.5)+320;
	float xOriMax =  int((tan(fieldOfView/2)*0.5)*320/2.5)+320;


	line(img, Point(xOriMin,480),Point(xMin*320/2.5+320,0),Scalar::all(255),1,8,0);
	line(img, Point(xOriMax,480),Point(xMax*320/2.5+320,0),Scalar::all(255),1,8,0);

	for(int i=0; i<tracker.peopleTracks.size(); i++)
	{
		
		Point2d center = Point2d(tracker.peopleTracks[i].prediction.x.ptr<double>(0)[0]*1000,tracker.peopleTracks[i].prediction.x.ptr<double>(0)[1]*1000);
		center.x = (center.x*0.001)*320/2.5+320;
		center.y = 480-(center.y*0.001)*240/3;

		/*Point2d arrow;
		arrow.x = center.x + tracker.peopleTracks[i].avgSpeed.x*1000;
		arrow.y = center.y +tracker.peopleTracks[i].avgSpeed.y*1000;
		arrow.x = (arrow.x*0.001)*320/2.5+320;
		arrow.y = 480-(arrow.y*0.001)*240/3;
		line(img, center, arrow, Scalar(0,0,255),2,8,0);*/
	
		if(tracker.peopleTracks[i].id == 0)
		{
			int sumX=0;
			int sumY=0;
			
			//getRobotLegPos(img,tracker.peopleTracks[i], Rinv, sumX, sumY);
			//cout<<sumY<<endl;
		
			boost::thread moveThread(boost::bind(&CCommand::moveContorl, &command,center.x, center.y, tracker.peopleTracks[i].lastSeen));
			moveThread.join();

			boost::thread rationThread(boost::bind(&CCommand::rationControl, &command,center.x, center.y, tracker.peopleTracks[i].lastSeen));
			rationThread.join();
		
		 }

		
		circle(img, center,4,Scalar(0,0,255),-1,8,0);
		char str[256];
		sprintf(str,"%d",tracker.peopleTracks[i].id);
		putText(img,str,center,CV_FONT_HERSHEY_COMPLEX, 1, /*tracker.peopleTracks[i].clr*/Scalar(0,0,255) );

	}
	imshow("people",img);
	
	
	Mat out;
	//resize(rgb, out, Size(640,480));
	cPointCloud.plotPC(Rinv,out,tracker.peopleTracks);
	imshow("rgb",out);
	

	waitKey(10);
	
}

void CFollowerRobot::getRobotLegPos(Mat & img, PeopleTracks peopleTracks,Mat Rinv, int& sumX, int &sumY)
{
	int sizeX = 640;
	int sizeY = 480;

	float fieldOfViewH = (58*1.0)/180*3.14;
	float fieldOfViewV = (45*1.0)/180*3.14;

	float fx = 320/tan(fieldOfViewH/2)*0.88;
	float fy = 240/tan(fieldOfViewV/2)*0.86;
	//camera extrinsic parameters
	Mat K = (Mat_<double>(3,3)<<fx, 0, sizeX/2,0, fy, sizeY/2, 0,0,1);
	Mat P = Rinv.rowRange(0,3).clone();
	Mat R = K*P;
	vector<Point2f>pts;
	for(int j=0; j<peopleTracks.legs.size(); j++)
	{
		double cx = peopleTracks.prediction.x.ptr<double>(0)[0]*1000;
		double cz = peopleTracks.prediction.x.ptr<double>(1)[0]*1000;
		Mat points = peopleTracks.legs[j].allPoints;		
		for(int m=0; m<points.rows; m++)
		{
			for(int n=0; n<points.cols; n++)
			{
				Point3d pt;
				pt.x = R.at<double>(0,0)*points.at<Vec3d>(m,n)[0] + R.at<double>(0,1)*points.at<Vec3d>(m,n)[1] + R.at<double>(0,2)*points.at<Vec3d>(m,n)[2] + R.at<double>(0,3);
				pt.y = R.at<double>(1,0)*points.at<Vec3d>(m,n)[0] + R.at<double>(1,1)*points.at<Vec3d>(m,n)[1] + R.at<double>(1,2)*points.at<Vec3d>(m,n)[2] + R.at<double>(1,3);
				pt.z = R.at<double>(2,0)*points.at<Vec3d>(m,n)[0] + R.at<double>(2,1)*points.at<Vec3d>(m,n)[1] + R.at<double>(2,2)*points.at<Vec3d>(m,n)[2] + R.at<double>(2,3);
				
				double imgX =  R.at<double>(0,0)*cx+R.at<double>(0,1)*7.5 + R.at<double>(0,2)*cz + R.at<double>(0,3);
				double imgY =  R.at<double>(1,0)*cx+R.at<double>(1,1)*7.5 + R.at<double>(1,2)*cz + R.at<double>(1,3);
				double imgZ =  R.at<double>(2,0)*cx+R.at<double>(2,1)*7.5 + R.at<double>(2,2)*cz + R.at<double>(2,3);
				int vx =cvRound( sizeX - imgX/imgZ);
				int vy =cvRound( sizeY - imgY/imgZ);

				int v = cvRound( sizeX - pt.x/pt.z );
				int u = cvRound( sizeY - pt.y/pt.z);

				if( v<=0||v>sizeX ||u<=0||u>sizeY)
					continue;
		
				pts.push_back(Point2f(v, u));
							
			}
		}
			
	}

	if(pts.size() == 0){ 
		return;
	}
	//cout<<"Not Seen !!!!"<<peopleTracks.lastSeen<<endl;
	RotatedRect box = minAreaRect(Mat(pts));
	Rect brect = box.boundingRect();
	sumX = box.boundingRect().x + box.boundingRect().width/2;
	sumY = box.boundingRect().y + box.boundingRect().height/2;
	//rectangle(img, brect, Scalar(0,0,255),1,8,0);
	circle(img, Point(sumX, sumY),5,Scalar(0,0,255),-1,8,0);
	//Point2f vertices[4];
	//box.points(vertices);
	//for (int i = 0; i < 4; i++)
	//		line(img, vertices[i], vertices[(i+1)%4], Scalar(0,0,255),2,8,0);

}
void CFollowerRobot::setTrackPeopleId(int id)
{   
	
}
void CFollowerRobot::robotFollower()
{
	
}