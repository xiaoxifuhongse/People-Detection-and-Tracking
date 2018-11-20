#pragma once
#include "utils.h"
#include <omp.h>

class CLegTracker
{
public:
	CLegTracker();
	~CLegTracker();

protected:
	void KFinitialize(Point2d initialPosition, double initialLegProbability, States& prediction);
	void KFpredict(States& previousPrediction, double sigmaAcc, double DT);// States prediction
	void PDAFupdate(States &prediction, Mat measures, double sigmaZ, double sigmaP, Mat& inGate);
	
	void getVecOfPeople(vector<PeopleTracks>&peopleTracks);
	template<typename T> bool ismember(T value,vector <T> vecs);
	template<typename T> void _find(vector<T>vecs, T val, int equ, vector<int>&res);
	template<typename T> int _find(vector<T> ves, T val);
    void transferMatToVecVec(Mat matrix, vector<Point2d>&vec);

	void _min(Mat dist, vector<double>&minVals, vector<int>&minIDs);
	int _min(vector<double>_dist,double &minVal);
	void pdist2(vector<Point2d>centroids1, vector<Point2d>centroids2, Mat &dist);
	void pdist2(Point2d centroids1,vector<Point2d>centroids2, vector<double>&dist);
	//void setdiff(vector<int>A, vector<int>B, vector<int>C);
	
	Mat delMat(Mat matrix, int row, int col);
	Mat delMatRow(Mat matrix , int row);
	Mat delMatCol(Mat matrix, int col);

	double _minDist(Mat allPoints1, Mat allPoints2, double minThreshold);
	int  nnz(vector<int>idxTracksToKill);
	int nnz(vector<bool>ves);

	vector<int> cat(vector<LegTracks>tracks, vector<int>idxTracks);
	vector<bool> _not(vector<bool>ves);

	// for people tracking
protected:
	void excludeLowProbabilityLegs(vector<LegTracks>&legTracks,vector<int>&candidatesTrack, vector<Candidate>&candidates);
	void predictPeopleTracks(vector<PeopleTracks> &peopleTracks, robot Tracker);

	void excludeLowProbabilityPeople(Mat normProbEstimages, 
									vector<pair<int, int>>&possibleLegsAssociation,
									vector<Point2d>&allPossiblePeopleCentroids,
									vector<double>&allPossiblePeopleProbEstimates);

	void removeTooClosePeople(vector<PeopleTracks>&peopleTracks);

	void getPossibleLegsAssociation(vector<Candidate>candidates,
									vector<LegTracks> &legTracks, 
									vector<int> &candidatesTrack,Mat legCentroids,
									vector<int>&isMatch, 
									vector<pair<int, int>> &possibleLegsAssociation );
	Mat getDistMatrix(vector<NewPeople>measures, vector<PeopleTracks>peopleTracks);

	void deletePossibleAssociation(vector<PeopleTracks>peopleTracks,vector<LegTracks>legTracks,vector<pair<int, int>>&possibleLegsAssociation,vector<Point2d>&allPossiblePeopleCentroids, vector<double>&allPossiblePeopleProbEstimates);
public:
	void candidatesTrackerPDAF(vector<Candidate>candidates, Mat probEstimates, robot &Tracker, vector<int>&candidateTracks);
	void peopleTrackerPDAFOnLegTracks(vector<Candidate>candidates, vector<int>candidatesTrack, robot& tracker);
	
	//----candidates legs tracking-------
private:
	
	int num_frames_not_seen_leg;
	double distanceThresholdForStraightAssociation;
	double distanceThresholdForNewTrackCreation;
	double distanceThresholdToOtherCandidates;
	double distanceThresholdForAssociationMultiple;
	double probThresholdNewCreation;
	
	
	int freeLegID;
	
	//---people tracking parameters--
private: 
	int num_freames_not_seen_people;
	int windowSizeSecond;
	int freePeopleID;

	RNG rng;
	double thresholdDistToOtherCentroid;
	double thresholdDist2LegsNewPeople; 
	double thresholdDist2LegsAssociation; 
	double thresholdDistStraightAssociation;
	double thresholdDistTooClosePeople; 
	double thresholdDistOverlappingCandidates; 
	double probabilityThresholdForAssociation;
	double probabilityDeviationThresholdForAssociation;
	double radiusPDAF;
	double thresholdStillPerson; 
	vector<PeopleTracks>candidatesPeople;
};