#pragma once
#include <opencv2/opencv.hpp>
#include <XnCppWrapper.h>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <omp.h>
#include <stdint.h>
#include <iterator> 
#include "nanoflann.hpp"


using namespace std;
using namespace cv;
using namespace nanoflann;
#ifdef _DEBUG
#define LINK_MULTI(a)      #a##"249d.lib"
#pragma comment(lib,LINK_MULTI(opencv_core))
#pragma comment(lib,LINK_MULTI(opencv_highgui))
#pragma comment(lib,LINK_MULTI(opencv_video))
#pragma comment(lib,LINK_MULTI(opencv_imgproc))
#pragma comment(lib,LINK_MULTI(opencv_features2d))
#pragma comment(lib,"openNI.lib")
#else
#define LINK_MULTI(a)      #a##"249.lib"
#pragma comment(lib,LINK_MULTI(opencv_core))
#pragma comment(lib,LINK_MULTI(opencv_highgui))
#pragma comment(lib,LINK_MULTI(opencv_video))
#pragma comment(lib,LINK_MULTI(opencv_imgproc))
#pragma comment(lib,LINK_MULTI(opencv_features2d))
#pragma comment(lib,"openNI.lib")
#endif

#define MAX_THREAD   32     // max thread
#define DIM          3    

//Marco
#define LEG_CUT_HEIGHT 500
#define DBSCAN_RADIUS  35
#define DBSCAN_NEIGH    3

#define SWIDTH  400
#define SHEIGHT 500
#define SDEPTH  600
#define SSIDE   10

#define V_PATCH_SIZE 10
#define H_PATCH_SIZE 10
#define NBINS         9
#define SDEPTH       600 //maximum depth

/*
Plane Parmeters£º aX+bY+cZ+d = 0
  
*/

class PlaneParmeters
{
public:
	PlaneParmeters()
	{ 
		a = 0.0;
		b = 0.0;
		c = 0.0;
		d = 0.0;
	};

	~PlaneParmeters(){};
	void logOut()
	{
		cout<<a <<" "<<b<<" "<<c<<" "<<d<<endl;
	}
	void operator = (PlaneParmeters other)
	{
		a = other.a;
		b = other.b;
		c = other.c;
		d = other.d;
	}
	void operator /=(double val)
	{
		a /= val;
		b /= val;
		c /= val;
		d /= val;
	}
	PlaneParmeters operator *(double val)
	{
		this->a =this->a * val;
		this->b =this->b * val;
		this->c =this->c * val;
		this->d =this->d * val;
		return *this;
	}

public :
	double a;
	double b;
	double c;
	double d;

};

struct PointCloud
{    
	double *pts;
	size_t numPoints;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const 
	{ 
		return numPoints;
	}
	// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
	inline double kdtree_distance(const double *p1, const size_t idx_p2, size_t size) const
	{
		const double d0 = p1[0] - pts[idx_p2*3];
		const double d1 = p1[1] - pts[idx_p2*3+1];
		const double d2 = p1[2] - pts[idx_p2*3+2];
		return (d0*d0) + (d1*d1) + (d2*d2);
	}

	// Returns the dim'th component of the idx'th point in the class:	
	inline double kdtree_get_pt(const size_t idx, int dim) const
	{
		return pts[idx*3+dim];
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX &bb) const { return false; }

};

/*
Candidate Object 
*/

struct Candidate
{
	Point2f rectangleOnFloor[4]; // vertices of the rectangle contationg the foot candidate projection onto the floor

	Mat footPoints;            //3xF matrix of 3D foot points(foot level)of the candidate

	Mat legPoints;             //3xL matrix of 3D leg points (from floor to knee level) of the candidate  

	Mat allPoints;             //3xA matrix of 3D whole points (from floor upwards until possible) of the candidates
	Mat legPointsNormalized;  // leg points rotated and translated
	Mat legColors;             // 3xL matrix of rgb colors corresponding to legPoints 
	Mat footColors;            // 3xF matrix of rgb colors corresponding to footPoints

	Point centroid;         //2D centroid of footPoints projected on floor
	Mat silhouette;         //silhouette of the candidate. It's the matrix containing the rectified depth image of the candidate leg


};


typedef struct _States
{
	Mat x;
	Mat P;

}States;

typedef struct _LEG_TRACKS
{
	States prediction;
	bool isOccupyed;
	int lastSeen;
	int id;
	_LEG_TRACKS()
	{
		lastSeen = 0;
		isOccupyed = false;
		id = -1;
	}

}LegTracks;

typedef struct _PEOPLE_TRACKS
{
	States prediction;
	int lifeTime;
	int lastSeen;
	int id;
	double heading;
	Point2d avgSpeed;
	vector<Candidate>legs;
	Mat vecSpeed;
	_PEOPLE_TRACKS()
	{
		avgSpeed = Point2d(0,0);
		heading = 0;
		lastSeen = 0;
		lifeTime = 0;
		id = 0;
		vecSpeed = Mat::zeros(2,30, CV_64FC1);
	}
	Scalar clr;


}PeopleTracks;


typedef struct _ROBOT
{
	double legSigmaZ;
	double legSigmaP;
	double legSigmaAcc;

	double peopleSigmaZ;
	double peopleSigmaP;
	double peopleSigmaAcc;
	double peopleDistThreshold;
	double legProbabilityThreshold;
	double refreshIntervalFloorPlane;
	double floorPlaneTolerance;

	vector<LegTracks>legTracks;
	vector<PeopleTracks>peopleTracks;
	int legFreeID;
	int peopleFreeID;
	double oldTimestamp;
	double currentTimestamp;
	PlaneParmeters floorPlane;
	Point3d pose;
	double floorPlaneTimestamp;

	vector<Candidate>candidates;

}robot;

typedef struct _NEW_PEOPLE
{
	Point2d centroid;
	double prob;
	vector<Candidate>legs;
	int lLegs;
	int rLegs;
	_NEW_PEOPLE()
	{
		lLegs = -1;
		rLegs = -1;
	}



}NewPeople;