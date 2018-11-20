#pragma once

#include "utils.h"
#include "libSVM.h"

class CPointCloud
{
public:
	/**
	 * @brief Create PointClound
	*/
	 CPointCloud();

	 /**
	  * @brief destroy the class
	 */
	 ~CPointCloud();


public: 
	/**
	 *  @brief voxel grid filter point cloud
	 *  @param[in]  matDepth  the depth image
	 *  @param[in]  matImage  the rgb image
	*/
	void getVoxelGrid(Mat matDepth, Mat matImage);

	/**
	 * @brief estimate the ground plane
	 * @param[in] best_plane extimate ground plane
	 * @param[in] maxInclinationAngle  maximum inclination angle that the camera
	 * @param[in] tol tolerance to be used to test when a point belongs to a plane
	*/
	bool getGroundPlane(PlaneParmeters& best_plane, float maxInclinationAngle , float tol );

	/**
	 * @brief rotate point to fix floor on the xy plane  
	 * @param[in] floorParmeters 
	 * @param[out] matInv for inverse transform
	*/
	void rotatePointCloud(PlaneParmeters floorParmeters,  Mat& matInv);

	void plotPC(Mat inv, Mat& imgOut,vector<PeopleTracks>peopleTracks);

	/**
	 * @brief get the candidate legs
	*/
	void getCandidateLegs(vector<Candidate>&cadidates);

	void predictClass(vector<Candidate>cadidates, Mat &prob);

protected:
	
	void hog(double *silhouette, double *features, int nPatchV, int nPatchH, int srow, int scol);

	void extractFeaturesHOG(vector<Candidate>candidates, Mat &features);

	void dbscanClustering(Mat footCutPoints, int nNeigh, int radius, vector<int>&classes, vector<int>& types);

	void expandCandidates(vector<Candidate>&candidates, Mat centroidOnFloorM, Mat voxelGridPoints, Mat voxelGridColors);

	void expandCandidate(double *footPoints, double *legPoints, double *personPoints, double *centroidsOnFloor,
						 KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud>, PointCloud, 3> *tree, 
						 size_t numFootPoints,
						 size_t *actualNumLegPoints, 
						 size_t *actualNumPersonPoints, 
						 size_t numCentroidsOnFloor, 
						 uint8_t *visited,
						 double *points, 
						 size_t idCandidate, 
						 uint8_t *colors,
						 uint8_t *legColors,
						 uint8_t *footColors, 
						 double *legPointsNormalized, 
						 double *silhouette, 
						 int numPoints); 
	void rotationMatrixFromAxisAndAngle(int axisX, int axisY, int axisZ, float angle, Mat& ratationMatrix);

	void minBoundingBox(vector<Point3d>points, Point2f vertices[]);

	float deg2rad(float angleInDergees);

	int  nnz(double *pointsPtr, PlaneParmeters plane, float tol);

private:


	double  *pointsPtr;
	uint8_t *colorsPtr;
	double  *rotatePointsPtr;

	Mat invM;
	int num_after_voxel;
	int rows;
	int cols;
	int numPoints;
	CSVM libSvm;


};