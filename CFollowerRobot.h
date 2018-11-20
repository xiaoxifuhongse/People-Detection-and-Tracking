#pragma once
#include "command.h"
#include "libSvm.h"
#include "CLegTracker.h"
#include "CPointCloud.h"

  
class CFollowerRobot
{
	
public: 
	CFollowerRobot();
	~CFollowerRobot(){};
public:
	
	void setupTracker();
	void trackPeople();
	void robotFollower();
	void setTrackPeopleId(int id);
	bool haveID;
	int followID;
protected:
	void update();
	void getRobotLegPos(Mat& img,PeopleTracks  candidate, Mat R, int& sumX, int &sumY);
private:
	CPointCloud cPointCloud;
	CLegTracker cTracks;

	robot tracker;
	xn::Context           g_Context;
	xn::DepthGenerator 	  g_DepthGenerator;
	xn::ImageGenerator    g_Image;

	xn::DepthMetaData     depthMD;
	xn::ImageMetaData     imageMD;
	int width;
	int height;
	Mat rgb;
	Mat depth;
	CCommand command;
	
};