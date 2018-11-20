#include "CLegTracker.h"
#include "CPointCloud.h"
#define SHOW 0
vector<PeopleTracks>hyposTracks;
CLegTracker::CLegTracker()
{

	num_frames_not_seen_leg = 3;
	distanceThresholdForStraightAssociation = 50;  // mm
	distanceThresholdForNewTrackCreation = 150;    // mm
	distanceThresholdToOtherCandidates = 100;      // mm
	distanceThresholdForAssociationMultiple = 100; //mm

	windowSizeSecond = 1;
		
	num_freames_not_seen_people  = 50; // maximum life of a not updated people track
	thresholdDistToOtherCentroid = 450; 
	thresholdDist2LegsNewPeople  = 100; 
	thresholdDist2LegsAssociation = 800; 
	thresholdDistTooClosePeople = 50; 
	thresholdDistOverlappingCandidates = 50; 
	probabilityDeviationThresholdForAssociation = 0.15;
	radiusPDAF = 600; 
	thresholdStillPerson = 0.05*0.05; 

}

CLegTracker::~CLegTracker()
{

}


/************************************************************************/
/* @brief tracking of all candidate legs
   @param[in] candidates = N candidate struct
   @param[in] probEstimates = Nx1 matrix containing the probability of each candidate 
                              to be a leg
  
/************************************************************************/

void CLegTracker::candidatesTrackerPDAF(vector<Candidate>candidates, Mat probEstimates, robot &tracker, vector<int>&candidateTracks)//
{
	
	vector<LegTracks>tracks;
	tracks    = tracker.legTracks;
	freeLegID = tracker.legFreeID;
	 
	double sigmaZ   = tracker.legSigmaZ;
	double sigmaP   = tracker.legSigmaP;
	double sigmaAcc = tracker.legSigmaAcc;

	//compute time elapsed from last update
	double DT = (tracker.currentTimestamp - tracker.oldTimestamp)/(double)getTickFrequency();
	//DT = 0.003;
	//predict step of Kalman Filter
	vector<LegTracks>::iterator iter = tracks.begin();
	while (iter!=tracks.end())
	{
		(*iter).lastSeen = (*iter).lastSeen + 1;
		if((*iter).lastSeen > num_frames_not_seen_leg)
		{
			iter = tracks.erase(iter);
		}
		else
		{
			KFpredict((*iter).prediction, sigmaAcc, DT);
			iter++;
		}	
	}
	if(candidates.size() == 0)
	{
		return;
	}
	//retrieve robot odometry for computing absolute measures
	double yaw  = tracker.pose.z;
	double cyaw = cos(yaw);
	double syaw = sin(yaw);

	// get the candidates centroid relative to robot
	vector<Point2d>centroids;
	vector<Point2d>trackCentroidsRelative;
	for(int i=0; i<candidates.size(); i++)
	{
		Point2d centroid = candidates[i].centroid;
		centroid.x       = cyaw*centroid.x  + syaw*centroid.y - tracker.pose.x;
		centroid.y       = -syaw*centroid.x + cyaw*centroid.y - tracker.pose.y;
		centroids.push_back(centroid);
	}

	// get the track centroids relative to robot
	for(int m=0; m<tracks.size(); m++)
	{
		Mat predictionX        = tracks[m].prediction.x;
		Point2d trackCentroids = Point2d(predictionX.at<double>(0,0)*1000, predictionX.at<double>(1,0)*1000);
		trackCentroids.x       = (cyaw*trackCentroids.x)  + (syaw*trackCentroids.y) - (tracker.pose.x);
		trackCentroids.y       = (-syaw*trackCentroids.x) + (cyaw*trackCentroids.y) - (tracker.pose.y);
		trackCentroidsRelative.push_back(trackCentroids);
	}

	vector<LegTracks> newTracks;  
	vector<int>       updatedTracks(tracks.size());
	Mat candidatesTracksProximity = Mat::zeros(candidates.size(), tracks.size(), CV_8UC1);

	//directly associate a candidate to a track if candidate foot covers track's centroid and then update 
	for(int i=0; i<candidates.size(); i++)
	{  
		vector<double> minDistanceMatrixs;
		vector<int>    idxTracks;
		double minMinDistanceMatrix = FLT_MAX;

		if(tracks.size()>0)
		{
			for(int m=0; m<tracks.size(); m++)
			{
				//track  cover candidates footpoints
				double *ptrFootPoints    = (double*)candidates[i].footPoints.data;
				double minDistanceMatrix = FLT_MAX;

				for(int n=0; n<candidates[i].footPoints.rows; n++)
				{
					double cx   = ptrFootPoints[3*n + 0];
					double cy   = ptrFootPoints[3*n + 1];
					double dist = sqrt((cx-trackCentroidsRelative[m].x)*(cx-trackCentroidsRelative[m].x) +
										(cy - trackCentroidsRelative[m].y)*(cy -trackCentroidsRelative[m].y));	

					if(dist < minDistanceMatrix)
					{
						minDistanceMatrix = dist;
					}
				}// for n
				
				minDistanceMatrixs.push_back(minDistanceMatrix);
				if(minMinDistanceMatrix > minDistanceMatrix)
				{
					minMinDistanceMatrix = minDistanceMatrix;
				}
				if(minDistanceMatrix <= distanceThresholdForStraightAssociation)
				{	
					idxTracks.push_back(m);
				}
			} //for tracks		
		}// if

		if(idxTracks.size() ==  1)// 1 candidates is straight associated to 1 track
		{
			Mat measures			   = Mat::zeros(1,3,CV_64FC1);
			measures.ptr<double>(0)[0] = centroids[i].x;
			measures.ptr<double>(0)[1] = centroids[i].y;
			measures.ptr<double>(0)[2] = 2*probEstimates.ptr<double>(1)[i]-1;

			PDAFupdate(tracks[idxTracks[0]].prediction,measures, sigmaZ, sigmaP, Mat());

			tracks[idxTracks[0]].lastSeen = 0;	
			updatedTracks[idxTracks[0]]   = 1;
			candidateTracks[i] = (idxTracks[0]);
		}
		else if(idxTracks.size() >= 2) // 1 candidate cover 2 tracks - update 2 tracks
		{
			// 1 candidate covers 2 tracks - update 2 tracks
			int idxTrack1, idxTrack2;
			int lastSeen1 = tracks[idxTracks[0]].lastSeen;
			int laseSeen2 = tracks[idxTracks[1]].lastSeen;

			if(lastSeen1 > laseSeen2)
			{
				idxTrack1 = idxTracks[1];
				idxTrack2  = idxTracks[0];
			}
			else
			{
				idxTrack1 = idxTracks[0];
				idxTrack2 = idxTracks[1];
			}

			Mat trackCentroididx1 = tracks[idxTrack1].prediction.x;
			Mat trackCentroididx2 = tracks[idxTrack2].prediction.x;

			Point2d trackCentroidsRelativeidx1 = Point2d(trackCentroididx1.at<double>(0,0)*1000, trackCentroididx1.at<double>(1,0)*1000);
			Point2d trackCentroidsRelativeidx2 = Point2d(trackCentroididx2.at<double>(0,0)*1000, trackCentroididx2.at<double>(1,0)*1000);
			
			Point2d jointCentroid ;
			jointCentroid.x = (trackCentroidsRelativeidx1.x + trackCentroidsRelativeidx2.x )/2;
		    jointCentroid.y = (trackCentroidsRelativeidx1.y + trackCentroidsRelativeidx2.y )/2;
			Point2d candidateCentroid = centroids[i];
			Point2d shiftVector;
			shiftVector.x = candidateCentroid.x -jointCentroid.x;
			shiftVector.y = candidateCentroid.y -jointCentroid.y;
			
			//track1
			Mat measures1 = Mat::zeros(1,3,CV_64FC1);
			measures1.ptr<double>(0)[0] = trackCentroidsRelativeidx1.x + shiftVector.x;
			measures1.ptr<double>(0)[1] = trackCentroidsRelativeidx1.y + shiftVector.y;
			measures1.ptr<double>(0)[2] = 2*probEstimates.ptr<double>(1)[i]-1;
			PDAFupdate(tracks[idxTrack1].prediction, measures1,sigmaZ, sigmaP,Mat());
			tracks[idxTrack1].lastSeen = 0;

			// track2
			Mat measures2 = Mat::zeros(1,3,CV_64FC1);
			measures2.ptr<double>(0)[0] = trackCentroidsRelativeidx2.x + shiftVector.x;
			measures2.ptr<double>(0)[1] = trackCentroidsRelativeidx2.y + shiftVector.y;
			measures2.ptr<double>(0)[2] = 2*probEstimates.ptr<double>(1)[i]-1;
			tracks[idxTrack2].lastSeen = 0;
			PDAFupdate(tracks[idxTrack2].prediction, measures2, sigmaZ, sigmaP, Mat());
			candidateTracks[i] = idxTrack1;

			updatedTracks[idxTrack1] = 1;
			updatedTracks[idxTrack2] = 1;
		}
		else
		{
			//candidatesTracksProximity(i, minDistanceMatrix < distanceThresholdForNewTrackCreation) = 1;
			for(int p=0; p<minDistanceMatrixs.size(); p++)
			{
				if(minDistanceMatrixs[p]<distanceThresholdForNewTrackCreation)
				{
					candidatesTracksProximity.ptr<uchar>(i)[p] = 1;
				}
			}

			//candidates not associate to any of existing tracks
			double minDistanceToOtherCandidates = FLT_MAX;
			for(int k=0; k<candidates.size(); k++)
			{
				if(k == i) continue;
	
				double distanceToOtherCandidates = sqrt((centroids[k].x - centroids[i].x)*(centroids[k].x - centroids[i].x) + (centroids[k].y - centroids[i].y)*(centroids[k].y - centroids[i].y));
			
			    if(minDistanceToOtherCandidates > distanceToOtherCandidates)
				{
					minDistanceToOtherCandidates = distanceToOtherCandidates;
				}
			}
			
			if(minMinDistanceMatrix> distanceThresholdForNewTrackCreation && minDistanceToOtherCandidates>distanceThresholdToOtherCandidates)
			{
				LegTracks newTrack;	
				KFinitialize(centroids[i], probEstimates.ptr<double>(1)[i],newTrack.prediction);
				Mat measures = Mat::zeros(1,3,CV_64FC1);
				measures.ptr<double>(0)[0] = centroids[i].x;
				measures.ptr<double>(0)[1] = centroids[i].y;
				measures.ptr<double>(0)[2] = 2*probEstimates.ptr<double>(1)[i]-1;
				
				PDAFupdate(newTrack.prediction, measures, sigmaZ,sigmaP,Mat());
				newTrack.lastSeen = 0;
				//newTrack.id       = freeLegID;
				//freeLegID += 1;
				newTracks.push_back(newTrack);   
				candidateTracks[i] = tracks.size() + newTracks.size()-1;	
			}

		}//else
	}//for

	//update the remaining tracks using PDAF approach
	for(int j=0; j<tracks.size(); j++)
	{
		if(updatedTracks[j] == 0)
		{
			vector<int>candidateIdx;
			for(int i=0; i<candidatesTracksProximity.rows; i++)
			{
				if(candidatesTracksProximity.ptr<uchar>(i)[j] != 0)
				{
					candidateIdx.push_back(i);
				}
			}
			// update using only the measures in the neighborhood
			if(candidateIdx.size()>0)
			{
				for(int m=0; m<candidateIdx.size(); m++)
				{
					Mat measures = Mat::zeros(1,3,CV_64FC1);
					measures.ptr<double>(0)[0] = centroids[candidateIdx[m]].x;
					measures.ptr<double>(0)[1] = centroids[candidateIdx[m]].y;
					measures.ptr<double>(0)[2] = 2*probEstimates.ptr<double>(1)[candidateIdx[m]]-1;
					PDAFupdate(tracks[j].prediction, measures, sigmaZ, sigmaP, Mat());
					tracks[j].lastSeen = max(0, tracks[j].lastSeen - 1);
				}		
			}
		}
	}
	// try to associate to something the unassociated candidates
	if(tracks.size()>0)
	{
		for(int j=0; j<candidateTracks.size(); j++)
		{ 
			if(candidateTracks[j] == -1)
		    {
				vector<double>help_dist;
				pdist2(candidates[j].centroid,trackCentroidsRelative,help_dist);
			    double minVal;
				int minId = _min(help_dist, minVal);
			
				if(minVal<distanceThresholdForAssociationMultiple)
				{
					candidateTracks[j] = minId;
				}
		     }
		}
	}

	for(int i=0; i<newTracks.size(); i++)
	{
		tracks.push_back(newTracks[i]);
	}


	tracker.legTracks = tracks;
	tracker.legFreeID = freeLegID;
	
	
}
void CLegTracker::excludeLowProbabilityLegs(vector<LegTracks>&legTracks,vector<int>&candidatesTrack, vector<Candidate>&candidates)
{
	//exclude from further processing leg tracks with low leg probability
	vector<int> idxTracksToKill;
	vector<int> idxTracksToSave;
	double thresholdFilterLegs = 0.8;
	for(int i=0; i<legTracks.size(); i++)
	{	
		double probabilities = (legTracks[i].prediction.x.ptr<double>(4)[0]+1)/2;
		if(probabilities<thresholdFilterLegs)
			idxTracksToKill.push_back(i);
		else
			idxTracksToSave.push_back(i);
	}
	vector<int>idxCandidatesToKill;
	for(int i=0; i<candidatesTrack.size(); i++)
	{
		if(ismember(candidatesTrack[i], idxTracksToKill))
		{
			idxCandidatesToKill.push_back(i);
		}
		else
		{
			int nz = 0;
			for(int m=0; m<idxTracksToKill.size(); m++)
			{
				if(idxTracksToKill[m]<candidatesTrack[i])
				{
					nz++;
				}
			}	
			candidatesTrack[i] = candidatesTrack[i] - nz;
		}		
	}
	sort(idxTracksToKill.begin(), idxTracksToKill.end(),greater<int>());
	sort(idxCandidatesToKill.begin(), idxCandidatesToKill.end(), greater<int>());
	for(int i=0; i<idxTracksToKill.size(); i++)
	{
		vector<LegTracks>::iterator itr = legTracks.begin()+ idxTracksToKill[i];
		legTracks.erase(itr);
	}
	for(int i=0; i<idxCandidatesToKill.size(); i++)
	{
		vector<Candidate>::iterator candidatesIter = candidates.begin() + idxCandidatesToKill[i];
		candidatesIter = candidates.erase(candidatesIter);

		vector<int>::iterator candidatesTrackIter = candidatesTrack.begin() + idxCandidatesToKill[i];
		candidatesTrackIter = candidatesTrack.erase(candidatesTrackIter);
	}
}
void CLegTracker::predictPeopleTracks(vector<PeopleTracks> &peopleTracks, robot tracker)
{
	//compute time elapsed from the previous update
	double DT = (tracker.currentTimestamp - tracker.oldTimestamp)/(double)getTickFrequency();

	// predict step of kalman filter and delete people tracks not seen for num_frame_no_seen_people times
	vector<PeopleTracks>::iterator peopleIter = peopleTracks.begin();
	while (peopleIter!= peopleTracks.end())
	{
		(*peopleIter).lastSeen = (*peopleIter).lastSeen + 1;
		if((*peopleIter).lastSeen > 30)
		{
			//remove track
			peopleIter = peopleTracks.erase(peopleIter);
		}
		else
		{
			KFpredict((*peopleIter).prediction, tracker.peopleSigmaAcc, DT);
			peopleIter++;
		}
	}

}
void CLegTracker::getPossibleLegsAssociation(vector<Candidate>candidates,vector<LegTracks> &legTracks, vector<int> &candidatesTrack, Mat legCentroids,vector<int>&isMatch, vector<pair<int, int>>&possibleLegsAssociation )
{	
	
	for(int i=0; i<legTracks.size(); i++)
	{
		int idL = legTracks[i].id;
		if(idL == -1)
		{   
			double minDist = FLT_MAX;
			int matchID = -1;
			for(int j=i+1; j<legTracks.size(); j++)
			{
				if(legTracks[j].id>=0 ||isMatch[i]==0 ||isMatch[j] == 0) continue;
				Point2d legCentroid1 = Point2d(legCentroids.ptr<double>(i)[0],legCentroids.ptr<double>(i)[1]);
				Point2d legCentroid2 = Point2d(legCentroids.ptr<double>(j)[0],legCentroids.ptr<double>(j)[1]);
				double  dist         = sqrt((legCentroid1.x - legCentroid2.x)*(legCentroid1.x - legCentroid2.x) + 
											(legCentroid1.y - legCentroid2.y)*(legCentroid1.y - legCentroid2.y));
				if(dist<minDist)
				{
					minDist = dist;
					matchID = j;
				}
			}
			if(minDist<500)
			{
			
				possibleLegsAssociation.push_back(make_pair(i,matchID));
				isMatch[i]  = 0;
				isMatch[matchID] = 0;
			}
		}
		else
		{
			bool isCP = false;
			int matchID = -1;
			double minDist = FLT_MAX;
			for(int j=i+1; j<legTracks.size(); j++)
			{
				if(isMatch[i]==0 ||isMatch[j] == 0) continue;
				Point2d legCentroid1 = Point2d(legCentroids.ptr<double>(i)[0],legCentroids.ptr<double>(i)[1]);
				Point2d legCentroid2 = Point2d(legCentroids.ptr<double>(j)[0],legCentroids.ptr<double>(j)[1]);
				double  dist         = sqrt((legCentroid1.x - legCentroid2.x)*(legCentroid1.x - legCentroid2.x) + 
											(legCentroid1.y - legCentroid2.y)*(legCentroid1.y - legCentroid2.y));
				if(dist<minDist)
				{
					minDist = dist;
					matchID = j;
				}
				if(legTracks[j].id == idL &&dist<800)
				{
					
					isCP = true;
					possibleLegsAssociation.push_back(make_pair(i,j));
					isMatch[i]  = 0;
					isMatch[j] = 0;
				
				}
			}

			if(isCP == false && minDist<500 && legTracks[matchID].id<0)
			{
				
				possibleLegsAssociation.push_back(make_pair(i,matchID));
				isMatch[i]  = 0;
				isMatch[matchID] = 0;
			}

		}

	}
	
	////remove more legs association containing already straight associated legs
	//vector<bool>visitedAssociations(possibleLegsAssociation.size());
	//while(nnz(_not(visitedAssociations)))
	//{
	//	int i = _find(visitedAssociations, false);
	//	visitedAssociations[i] = 1;
	//	int legTrack1 = possibleLegsAssociation[i].first;
	//	int legTrack2 = possibleLegsAssociation[i].second;
	//	vector<int>idCandidates1,idCandidates2;
	//	_find(candidatesTrack, legTrack1, 0, idCandidates1);
	//	_find(candidatesTrack, legTrack2, 0, idCandidates2);
	//	if(idCandidates1.size()>0 && idCandidates2.size()>0)
	//	{
	//		
	//		double mindist = _minDist(candidates[idCandidates1[0]].allPoints, candidates[idCandidates2[0]].allPoints, thresholdDistOverlappingCandidates);
	//		if(mindist <= thresholdDistOverlappingCandidates )// ||
	//		{	
	//			vector<int>idxToKill;
	//			for(int m=0; m<possibleLegsAssociation.size(); m++)
	//			{
	//				if( possibleLegsAssociation[m].first == legTrack1 || possibleLegsAssociation[m].first == legTrack2 ||
	//					possibleLegsAssociation[m].second == legTrack1 || possibleLegsAssociation[m].second == legTrack2)
	//				{
	//					if(m != i)
	//					{
	//						idxToKill.push_back(m);
	//					}
	//				}
	//			}	
	//			sort(idxToKill.begin(), idxToKill.end(), greater<int>());
	//			for(int k=0; k<idxToKill.size(); k++)
	//			{
	//				vector<pair<int,int>>::iterator itr = possibleLegsAssociation.begin() + idxToKill[k];
	//				possibleLegsAssociation.erase(itr);
	//				vector<bool>::iterator itr1 = visitedAssociations.begin() + idxToKill[k];
	//				visitedAssociations.erase(itr1);
	//			}
	//		}
	//	}
	//}
}
void CLegTracker::excludeLowProbabilityPeople(Mat normProbEstimages, vector<pair<int, int>>&possibleLegsAssociation,vector<Point2d>&allPossiblePeopleCentroids, vector<double>&allPossiblePeopleProbEstimates)
{
	//remove associations with low leg probability
	vector<double>probNorm = Mat_<double>(normProbEstimages);
	vector<int>idxLegTracksToKill;
	for(int i=0; i<probNorm.size(); i++)
	{
		if(probNorm[i]<0.5)//
		{
			idxLegTracksToKill.push_back(i);	
		}
	}
	vector<int>idxToKill;
	for(int m=0; m<possibleLegsAssociation.size(); m++)
	{
		bool isMember = false;
		for(int n=0; n<idxLegTracksToKill.size(); n++)
		{
			if(possibleLegsAssociation[m].first == idxLegTracksToKill[n] || possibleLegsAssociation[m].second ==idxLegTracksToKill[n])
			{
				isMember = true;
			}
		}
		if(isMember)
			idxToKill.push_back(m);
	}
	sort(idxToKill.begin(), idxToKill.end(), greater<int>());
	for(int i=0; i<idxToKill.size(); i++)
	{
		vector<pair<int, int>>::iterator itr = possibleLegsAssociation.begin()+idxToKill[i];
		possibleLegsAssociation.erase(itr);
		vector<Point2d>::iterator itr1 = allPossiblePeopleCentroids.begin() + idxToKill[i];
		allPossiblePeopleCentroids.erase(itr1);
		vector<double>::iterator itr2 = allPossiblePeopleProbEstimates.begin() + idxToKill[i];
		allPossiblePeopleProbEstimates.erase(itr2);	 
	}
}
void CLegTracker::removeTooClosePeople(vector<PeopleTracks>&peopleTracks)
{
	////check if two people tracks are too close and delete the older one
	vector<Point2d>peopelTracksCentroids; // people centroids
	for(int i=0; i<peopleTracks.size(); i++)
	{
		peopelTracksCentroids.push_back(Point2d(peopleTracks[i].prediction.x.ptr<double>(0)[0]*1000, peopleTracks[i].prediction.x.ptr<double>(1)[0]*1000));
	}
	Mat peopleDistances; // distance matrix of two people centroids
	pdist2(peopelTracksCentroids, peopelTracksCentroids, peopleDistances);
	peopleDistances +=  Mat::eye(peopelTracksCentroids.size(),peopelTracksCentroids.size(),CV_64FC1)*FLT_MAX;
	vector<double>minVals;
	vector<int> minTracks2IDs;
	double minVal;
	_min(peopleDistances,minVals, minTracks2IDs);
	int track1 =_min(minVals, minVal);
	while(!peopleDistances.empty() && minVal<thresholdDistTooClosePeople)
	{
		int track2 = minTracks2IDs[track1];
		int trackToKill;
		if(peopleTracks[track1].lastSeen <= peopleTracks[track2].lastSeen)
		{
			trackToKill = track2;

		}
		else
			trackToKill = track1;

		vector<PeopleTracks>::iterator itr = peopleTracks.begin() + trackToKill;
		peopleTracks.erase(itr);
		vector<Point2d>::iterator itr1 = peopelTracksCentroids.begin() + trackToKill;
		peopelTracksCentroids.erase(itr1);
		peopleDistances = delMat(peopleDistances, trackToKill, trackToKill);
		minTracks2IDs.clear();
		minVals.clear();
		_min(peopleDistances,minVals, minTracks2IDs);
		track1 =_min(minVals, minVal);
	}
}
Mat CLegTracker::getDistMatrix(vector<NewPeople>measures, vector<PeopleTracks>peopleTracks)
{
	int m = measures.size();
	int n = peopleTracks.size();

	Mat distMatrix = Mat::zeros(m, n, CV_64FC1);
	Mat costMatrix = Mat::zeros(m, n,CV_8UC1);
	
	if(m == 0||n == 0)
	{
		return costMatrix;
	}
	for(int i=0; i<m; i++)
	{	
		for(int j=0; j<n; j++)
		{
			double mx = measures[i].centroid.x/1000;
			double my = measures[i].centroid.y/1000;

			double nx   = peopleTracks[j].prediction.x.ptr<double>(0)[0];
			double ny   = peopleTracks[j].prediction.x.ptr<double>(0)[1];

			double dx = abs(nx - mx);
			double dy = abs(nx - my);
			double sigma = 0.5;

			double dist = ((mx-nx)*(mx-nx)+(my-ny)*(my-ny));
			dist = exp((dist)/(2*sigma))/sqrt(2*3.14*sigma);
			distMatrix.ptr<double>(i)[j] = dist;
		}
	}

	for(int i=0; i<m; i++)
	{
		int id = 0;
		double fmin = FLT_MAX;
		for(int j=0; j<n;j++)
		{
			if(fmin>distMatrix.at<double>(i,j))
			{
				fmin = distMatrix.at<double>(i,j);
				id = j;
			}
		}
		if(fmin<0.8)
		{	
			costMatrix.at<uchar>(i,id) += 1;
		}
	}
	//cout<<costMatrix<<endl;
	for(int j=0; j<n; j++)
	{
		int id = 0;
		double fmin = FLT_MAX;

		for(int i=0; i<m; i++)
		{
			if(fmin>distMatrix.at<double>(i,j))
			{
				id = i;
				fmin = distMatrix.at<double>(i,j);
			}
		}

		if(fmin<0.8)
		{	
			costMatrix.at<uchar>(id,j) += 1;
		}
	}
	cout<<"---------------"<<endl;
	cout<<distMatrix<<endl;
	
	cout<<"+++++++++++++"<<endl;
	return costMatrix;
}
void CLegTracker::getVecOfPeople(vector<PeopleTracks>&peopleTracks)
{
	
	for(int i=0; i<peopleTracks.size(); i++)
	{
		int nLength = peopleTracks[i].vecSpeed.cols;
		Mat tmp = Mat::zeros(2,30,CV_64FC1);
		for(int m=0; m<peopleTracks[i].vecSpeed.rows; m++)
		{
			for(int n=1; n<peopleTracks[i].vecSpeed.cols; n++)
			{
				tmp.ptr<double>(m)[n] = peopleTracks[i].vecSpeed.ptr<double>(m)[n-1];
			}
		}
		double vx = peopleTracks[i].prediction.x.ptr<double>(2)[0];
		double vy = peopleTracks[i].prediction.x.ptr<double>(3)[0];
		tmp.ptr<double>(0)[0] = vx;
		tmp.ptr<double>(1)[0] = vy;
		tmp.copyTo(peopleTracks[i].vecSpeed);
	
		double sumX =0.0;
		double sumY = 0.0;
		for(int n=0; n<tmp.cols; n++)
		{
			sumX+= tmp.ptr<double>(0)[n];
			sumY+= tmp.ptr<double>(1)[n];

		}
		sumX=sumX/tmp.cols;
		sumY = sumY/tmp.cols;
		peopleTracks[i].avgSpeed.x =sumX;
		peopleTracks[i].avgSpeed.y =sumY;
		if((sumX*sumX+sumY*sumY) > thresholdStillPerson)
			peopleTracks[i].heading = atan2(peopleTracks[i].avgSpeed.y, peopleTracks[i].avgSpeed.x);
		
	}
}

/************************************************************************/
/*
	    People Tracker over the candidate track obtained by candidatesTrackerPDAF
   
   Inputs:
     candidates = Nx1 vector of candidate structs as obtained by getCandidateLegs
     candidatesTrack = Nx1 vector as obtained by candidatesTrackerPDAF. 
                       It contains the index of single leg track to which each 
                       candidate has been associated.
     tracker = tracker object containing the state of the single candidate tracker
               and of this people tracker.
     
   Outputs:
     legTrackOnPeople = Mx1 vector containing for each leg track the index
                        of the person to which it as been associated, or 0 if 
                        no association was possible
     tracker = updated tracker object
*/
/************************************************************************/

void CLegTracker::peopleTrackerPDAFOnLegTracks(vector<Candidate>candidates, vector<int>candidatesTrack, robot &tracker)
{
	
	thresholdDistStraightAssociation   = tracker.peopleDistThreshold;
	probabilityThresholdForAssociation = tracker.legProbabilityThreshold;
	vector<LegTracks>legTracks = tracker.legTracks;
	excludeLowProbabilityLegs(legTracks,candidatesTrack, candidates);//1
	
	///////////////////////////////////////////SETP 2/////////////////////////////////////////////////////////
	// read parmeters from tracker
	vector<PeopleTracks>peopleTracks;
	peopleTracks    = tracker.peopleTracks;
    freePeopleID    = tracker.peopleFreeID;
	double sigmaZ   = tracker.peopleSigmaZ;
	double sigmaP   = tracker.peopleSigmaP;
	double sigmaAcc = tracker.peopleSigmaAcc;
	getVecOfPeople(peopleTracks);
	predictPeopleTracks(peopleTracks, tracker);
	removeTooClosePeople(peopleTracks);

	//no leg tracks
	if(legTracks.size() == 0)
	{
		tracker.peopleFreeID = freePeopleID;
		tracker.peopleTracks = peopleTracks;
		return;
	}

	//compute feasible associations between leg tracks
	Mat legCentroids           = Mat::zeros(legTracks.size(), 2, CV_64FC1); // legTracks centroid
	Mat probEstimates          = Mat ::zeros(legTracks.size(),1, CV_64FC1); // probibality of legTracks
	Mat normProbEstimages      = Mat ::zeros(legTracks.size(),1, CV_64FC1); // norm probibality of legTracks
	Mat probEstimateDeviations = Mat ::zeros(legTracks.size(),1, CV_64FC1);
	for(int i=0; i<legTracks.size();i++)
	{
		legCentroids.ptr<double>(i)[0]      = legTracks[i].prediction.x.ptr<double>(0)[0]*1000;
		legCentroids.ptr<double>(i)[1]      = legTracks[i].prediction.x.ptr<double>(0)[1]*1000;
		probEstimates.ptr<double>(i)[0]     = legTracks[i].prediction.x.ptr<double>(0)[4];
		normProbEstimages.ptr<double>(i)[0] = (legTracks[i].prediction.x.ptr<double>(0)[4]+1)/2;
	}	

//	////////////////////////////////////////////////////////////////////////////////////////////
	vector<pair<int, int> >possibleLegsAssociation;
	vector<int>isMatch(legTracks.size(), -1);
//// need to fix 
	getPossibleLegsAssociation(candidates, legTracks, candidatesTrack, legCentroids, isMatch, possibleLegsAssociation);

	vector<Point2d>allPossiblePeopleCentroids;
	vector<double>allPossiblePeopleProbEstimates;
	for(int i=0; i<possibleLegsAssociation.size(); i++)
	{
		int m = possibleLegsAssociation[i].first;
		int n = possibleLegsAssociation[i].second;
		double x = (legCentroids.ptr<double>(m)[0]+legCentroids.ptr<double>(n)[0])/2;
		double y = (legCentroids.ptr<double>(m)[1]+legCentroids.ptr<double>(n)[1])/2;
		allPossiblePeopleProbEstimates.push_back((probEstimates.ptr<double>(m)[0] + probEstimates.ptr<double>(n)[0])/2);
		allPossiblePeopleCentroids.push_back(Point2d(x,y));
	}
	excludeLowProbabilityPeople(normProbEstimages,possibleLegsAssociation,allPossiblePeopleCentroids,allPossiblePeopleProbEstimates);
	/////////////////////////////////////////////////////////////////////////
	vector<NewPeople>newPeoples;
	for(int i=0; i<possibleLegsAssociation.size(); i++)
	{
		int m = possibleLegsAssociation[i].first;
		int n = possibleLegsAssociation[i].second;
	
		NewPeople newTrack;
		vector<int>id1;
		_find(candidatesTrack, m, 0, id1);
		if(id1.size() >0 )
		{
			newTrack.legs.push_back(candidates[id1[0]]);	
		}
		id1.clear();
		_find(candidatesTrack,n, 0,id1);
		if(id1.size()>0)
		{
			newTrack.legs.push_back(candidates[id1[0]]);
		}
		newTrack.lLegs = m;
		newTrack.rLegs = n;
		newTrack.centroid.x = allPossiblePeopleCentroids[i].x;
		newTrack.centroid.y = allPossiblePeopleCentroids[i].y;
		
		newTrack.prob = allPossiblePeopleProbEstimates[i];
		newPeoples.push_back(newTrack);
	}

	Mat cost = getDistMatrix(newPeoples, peopleTracks);
	

	cout<<cost<<endl;

	vector<int>update(peopleTracks.size());
	vector<PeopleTracks>help;
	for(int i=0; i<newPeoples.size(); i++)
	{
		int sum = 0;
		for(int j=0; j<peopleTracks.size();j++)
		{
			if(cost.at<uchar>(i,j) == 2) // update object
			{
				update[j] = 1;
				peopleTracks[j].lastSeen = 0;
				Mat measures = Mat::zeros(1,3,CV_64FC1);
				measures.ptr<double>(0)[0] = newPeoples[i].centroid.x;
				measures.ptr<double>(0)[1] = newPeoples[i].centroid.y;
				measures.ptr<double>(0)[2] = newPeoples[i].prob;
				peopleTracks[j].legs = newPeoples[i].legs;
				legTracks[newPeoples[i].lLegs].id = peopleTracks[j].id;
				legTracks[newPeoples[i].rLegs].id = peopleTracks[j].id;
				PDAFupdate(peopleTracks[j].prediction, measures, sigmaZ, sigmaP, Mat());
			}
			if(cost.at<uchar>(i,j) == 0)
			{
				sum++;
			}
		}
		if(sum == peopleTracks.size()) // new object
		{
			cout<<"new object"<<endl;
			PeopleTracks newTrack;
			KFinitialize(newPeoples[i].centroid, newPeoples[i].prob, newTrack.prediction);
			Mat measures = Mat::zeros(1,3,CV_64FC1);
			measures.ptr<double>(0)[0] = newPeoples[i].centroid.x;
			measures.ptr<double>(0)[1] = newPeoples[i].centroid.y;
			measures.ptr<double>(0)[2] = newPeoples[i].prob;
			PDAFupdate(newTrack.prediction, measures, sigmaZ, sigmaP, Mat());
			newTrack.id = freePeopleID;
			newTrack.legs = newPeoples[i].legs;
			newTrack.heading = 3*CV_PI/2;
			legTracks[newPeoples[i].lLegs].id = freePeopleID;
			legTracks[newPeoples[i].rLegs].id = freePeopleID;
			newTrack.avgSpeed = Point2d(newTrack.prediction.x.ptr<double>(0)[0],newTrack.prediction.x.ptr<double>(0)[1]);
			newTrack.clr = Scalar(rand()%255, rand()%255, rand()%255);
			freePeopleID++;
			help.push_back(newTrack);
		}

	}

   //one leg hyposis of a people
	for(int i=0; i<legTracks.size(); i++)
	{
		//if(isMatch[i] == -1 )//&&  probEstimates.ptr<double>(i)[0]>0.6
		{
			vector<int>idx;
			_find(candidatesTrack, i,0, idx);
			if(idx.size() == 0) continue;
			for(int j=0; j<update.size(); j++)
			{
				if(update[j] == 0) // no update
				{
					double mx  = legTracks[i].prediction.x.ptr<double>(0)[0];
					double my  = legTracks[i].prediction.x.ptr<double>(0)[1];

					double nx = peopleTracks[j].prediction.x.ptr<double>(0)[0];
					double ny = peopleTracks[j].prediction.x.ptr<double>(0)[1];
					double dist = (mx-nx) * (mx-nx) +(my-ny) *(my-ny);
				
					double dx = abs(nx - mx);
					double dy = abs(nx - my);
					double sigma = 0.5;
					dist = exp((dist)/(2*sigma))/sqrt(2*3.14*sigma);
					cout<<"!!!!!!!!!!!!!!!!!!   "<<dist<<endl;
					if(dist<0.7)//&&probEstimates.ptr<double>(i)[0]
					{
		
						//legTracks[i].id = peopleTracks[j].id;
						Mat measures = Mat::zeros(1,3,CV_64FC1);
						measures.ptr<double>(0)[0] = mx*1000;
						measures.ptr<double>(0)[1] = my*1000;
						measures.ptr<double>(0)[2] = probEstimates.ptr<double>(i)[0];
						peopleTracks[j].legs.clear();
						peopleTracks[j].legs.push_back(candidates[idx[0]]);
						PDAFupdate(peopleTracks[j].prediction, measures, sigmaZ, sigmaP, Mat());
						peopleTracks[j].lastSeen = 0;
					}	
					else
					{
						legTracks[i].id =-1;
					}
				}
			}
		}
	}
	for(int i=0; i<help.size(); i++)
		peopleTracks.push_back(help[i]);

	tracker.peopleTracks = peopleTracks;
	tracker.peopleFreeID = freePeopleID;	
	tracker.legTracks.clear();
	tracker.legTracks    = legTracks;
	
#if 1
	Mat img2 = Mat::zeros(Size(640,480),CV_8UC3);
	for(int i=0; i<isMatch.size(); i++)
	{
		if(isMatch[i] == -1 &&  probEstimates.ptr<double>(i)[0]>0.4 )
		{
			vector<int>idx;
			_find(candidatesTrack, i,0, idx);
			if(idx.size() == 0) continue;
			int id = i;
			float yMax = 3.5;
			float fieldOfView =(float)57/180*CV_PI;
			float xMin = -tan(fieldOfView/2)*yMax;
			float xMax = tan(fieldOfView/2)*yMax;
			float yMin = 0;


			float xOriMin = int((-tan(fieldOfView/2)*0.5)*320/2.5)+320;
			float xOriMax =  int((tan(fieldOfView/2)*0.5)*320/2.5)+320;


			line(img2, Point(xOriMin,480),Point(xMin*320/2.5+320,0),Scalar::all(255),1,8,0);
			line(img2, Point(xOriMax,480),Point(xMax*320/2.5+320,0),Scalar::all(255),1,8,0);


			Point2d center = Point2d(legTracks[id].prediction.x.ptr<double>(0)[0]*1000,legTracks[id].prediction.x.ptr<double>(0)[1]*1000);

			center.x = (center.x*0.001)*320/2.5+320;
			center.y = 480-(center.y*0.001)*240/3;

			circle(img2, center,10,Scalar(128,255,128),-1,8,0);
		}
	}


	for(int i=0; i<possibleLegsAssociation.size(); i++)
	{
		int id1 = possibleLegsAssociation[i].first;
		int id2 = possibleLegsAssociation[i].second;
		float yMax = 3.5;
		float fieldOfView =(float)57/180*CV_PI;
		float xMin = -tan(fieldOfView/2)*yMax;
		float xMax = tan(fieldOfView/2)*yMax;
		float yMin = 0;

		
		float xOriMin = int((-tan(fieldOfView/2)*0.5)*320/2.5)+320;
		float xOriMax =  int((tan(fieldOfView/2)*0.5)*320/2.5)+320;


		line(img2, Point(xOriMin,480),Point(xMin*320/2.5+320,0),Scalar::all(255),1,8,0);
		line(img2, Point(xOriMax,480),Point(xMax*320/2.5+320,0),Scalar::all(255),1,8,0);

		
		Point2d center = Point2d(legTracks[id1].prediction.x.ptr<double>(0)[0]*1000,legTracks[id1].prediction.x.ptr<double>(0)[1]*1000);

		center.x = (center.x*0.001)*320/2.5+320;
		center.y = 480-(center.y*0.001)*240/3;

		Point2d center1 = Point2d(legTracks[id2].prediction.x.ptr<double>(0)[0]*1000,legTracks[id2].prediction.x.ptr<double>(0)[1]*1000);
		center1.x = (center1.x*0.001)*320/2.5+320;
		center1.y = 480-(center1.y*0.001)*240/3;

		
		Point2d pCenter = allPossiblePeopleCentroids[i];
		pCenter.x = (pCenter.x*0.001)*320/2.5+320;
		pCenter.y = 480-(pCenter.y*0.001)*240/3;
		circle(img2, pCenter,8,Scalar(255,255,128),-1,8,0);
		circle(img2, center,5,Scalar(0,0,255),-1,8,0);
		circle(img2, center1,5,Scalar(0,0,255),-1,8,0);
		line(img2, pCenter, center1, Scalar(0,255,0),1,8,0);	
		line(img2, pCenter, center, Scalar(0,255,0),1,8,0);	
		}
		imshow("possible",img2);
#endif

	
}

void CLegTracker::deletePossibleAssociation(vector<PeopleTracks>peopleTracks,vector<LegTracks>legTracks, vector<pair<int, int>>&possibleLegsAssociation,vector<Point2d>&allPossiblePeopleCentroids, vector<double>&allPossiblePeopleProbEstimates)
{
	

	vector<int>avialbleLeg(legTracks.size());
	vector<int>totalUsed(legTracks.size());
	
	for(int i=0; i<possibleLegsAssociation.size(); i++)
	{
		//cout<<"possible num:"<<i<<endl;
		int m = possibleLegsAssociation[i].first;
		int n = possibleLegsAssociation[i].second;
		
		double centroidx = allPossiblePeopleCentroids[i].x;
		double centroidy = allPossiblePeopleCentroids[i].y;
		double minDist = FLT_MAX;
		for(int j=0; j<peopleTracks.size(); j++)
		{
			double mx = peopleTracks[j].prediction.x.ptr<double>(0)[0]*1000;
			double my = peopleTracks[j].prediction.x.ptr<double>(1)[0]*1000;
			double dist = sqrt((centroidx-mx)*(centroidx-mx) + (centroidy-my)*(centroidy-my));
			if(dist<FLT_MAX)
			{
				minDist = dist;
			}
		}
	
		if(minDist<100)
		{
			avialbleLeg[m] += 1;
			avialbleLeg[n] += 1;	
		}
		
	}

	vector<int>idxToKill;
	for(int i=0; i<possibleLegsAssociation.size(); i++)
	{
		int m = possibleLegsAssociation[i].first;
		int n = possibleLegsAssociation[i].second;

		int p = avialbleLeg[m];
		int q = avialbleLeg[n];

		if(p!=q ||p>1 ||q>1)
		{

			idxToKill.push_back(i);			
		}

	}

	sort(idxToKill.begin(), idxToKill.end(),greater<int>());
	for(int i=0; i<idxToKill.size(); i++)
	{
		vector<pair<int, int>>::iterator itr1 = possibleLegsAssociation.begin()+ idxToKill[i];
		possibleLegsAssociation.erase(itr1);
		vector<Point2d>::iterator itr2 = allPossiblePeopleCentroids.begin()+ idxToKill[i];
		allPossiblePeopleCentroids.erase(itr2);
		vector<double>::iterator itr3 = allPossiblePeopleProbEstimates.begin()+ idxToKill[i];
		allPossiblePeopleProbEstimates.erase(itr3);
	}
	
}
void CLegTracker::KFinitialize(Point2d initialPosition, double initialLegProbability, States &prediction)
{
	Mat P = (Mat_<double>(5,5)<<0.3, 0,   0,   0,  0,
		                        0,   0.3, 0,   0,  0,
								0,   0,   60,  0,  0,
								0,   0,   0,   60, 0,
								0,   0,   0,   0,  1);

	Mat x = (Mat_<double>(5,1)<<initialPosition.x/1000,
		                        initialPosition.y/1000,
								0,
								0,
								initialLegProbability);
	
	P.copyTo(prediction.P);
	x.copyTo(prediction.x);
}

/**
	 *  @brief apply prediction step of Kalman Filter
	 *  @param[in]  previousPrediction containing state prediction x and error covariance p
	 *  @param[in]  DT  the time interval between two measures
	*/

void CLegTracker::KFpredict(States &previousPrediction, double sigmaAcc, double DT)//States prediction
{
	Mat F = (Mat_<double>(5,5)<<1, 0, DT, 0,  0,
		                        0, 1, 0,  DT, 0,
								0, 0, 1,  0,  0,
								0, 0, 0,  1,  0,
								0, 0, 0,  0,  1);

	double G1 = (DT*DT)/2;
	double G2 = DT;
	
	double Q11 = G1*G1*(sigmaAcc*sigmaAcc); 
    double Q12 = G1*G2*(sigmaAcc*sigmaAcc);
	double Q21 = G2*G1*(sigmaAcc*sigmaAcc);
	double Q22 = G2*G2*(sigmaAcc*sigmaAcc);

	Mat Q = (Mat_<double>(5,5)<<Q11,   0, Q12,   0,  0,
		                          0, Q11,   0, Q12,  0,
								Q21,   0, Q22,   0,  0,
								  0, Q21,   0, Q22,  0,
								  0,   0,   0,   0,  0);

	if(previousPrediction.x.empty() && previousPrediction.P.empty())
	{
		double L = 1000000;
		previousPrediction.x = Mat::zeros(1,5,CV_64FC1);
		previousPrediction.P = Mat::eye(5,5,CV_64FC1)*L;
		return;
	}
	
	previousPrediction.x = F*previousPrediction.x;
	previousPrediction.P = F*previousPrediction.P*F.t() + Q;

}

void CLegTracker::PDAFupdate(States& prediction, Mat measures, double sigmaZ, double sigmaP, Mat &inGate)
{
	
	for(int i=0; i<measures.rows; i++)
	{
		measures.at<double>(i,0) = measures.at<double>(i,0)/1000;
		measures.at<double>(i,1) = measures.at<double>(i,1)/1000;
	}
	// observation model
	Mat H = (Mat_<double>(3,5)<<1, 0, 0, 0, 0,
		                        0, 1, 0, 0, 0,
							    0, 0, 0, 0, 1);
	//covariance of nosie on observation model
	Mat R = (Mat_<double>(3,3)<<sigmaZ*sigmaZ, 0, 0,
		                        0,sigmaZ*sigmaZ,  0,
								0, 0, sigmaP*sigmaP);
	// innovation covariance
	Mat S = H*prediction.P*H.t() + R;
	
	//S = tril(S,-1)+tril(S)'; force simmetry of S
	Mat Spos = S(Range(0,2),Range(0,2));

	double c = CV_PI;
	int n = 2;  //dimension of measurements
	//measurement validation
	double gamma = 20;            //radius of gate region in meters
	double gammaNewTrack = 200;  // radius to create a new track in meters

	double V = c*pow(gamma, n/2)*sqrt(determinant(Spos));// volume of validation region
	
    int nMeasures = measures.rows;
	Mat zHat = H*prediction.x;
	
	Mat zHatPos = zHat.rowRange(0,2);
	
	Mat measuresValidated  = Mat::zeros(nMeasures, 1, CV_8UC1);
    inGate = Mat::zeros(nMeasures, 1, CV_8UC1);
	Mat createNewTrack = Mat::zeros(nMeasures, 1, CV_8UC1);

	for(int i=0; i<nMeasures; i++)
	{
		Mat z = measures(Range(i,i+1),Range(0,2));
		
		Mat a = (Spos.inv()*(z.t()-zHatPos)).t() *(z.t()-zHatPos);
		
		if(a.at<double>(0,0)<=gamma)
	    {
			measuresValidated.at<uchar>(0,i) = 1;
			inGate.at<uchar>(0,i) = 1;
		}
		else if(a.at<double>(0,0)>=gammaNewTrack)
		{
			createNewTrack.at<uchar>(0,i) = 1;
		}
	}//for
	Mat measures_resize;
	int m = countNonZero(measuresValidated);

	if (m == 0)
	{
		return;
	}
	measures_resize = Mat::zeros(m,3,CV_64FC1);
	int j=0;
	for(int i=0; i<measures.rows; i++)
	{
		if(measuresValidated.at<uchar>(0,i) == 1)
		{ 
			measures_resize.at<double>(j,0) = measures.at<double>(i,0);
			measures_resize.at<double>(j,1) = measures.at<double>(i,1);
			measures_resize.at<double>(j,2) = measures.at<double>(i,2);
			j++;
		}
	}
	
	////data association
	double Pg = 0.8;   //probability that gate contains the true measure if detected (corresponding to gamma)
	double Pd = 0.7;   //target detection probability

	////mvnpdf
	
	double alpha = pow((2*CV_PI), -(measures_resize.cols/2))* pow(determinant(Spos), -0.5);
	
	Mat likelihood = Mat::zeros(m,1,CV_64FC1);
	double sumLikelihood = 0;
	for(int i=0; i<m;i++)
	{
		Mat x = measures_resize(Range(i,i+1),Range(0,2));;
		Mat diff = x.t()-zHatPos;
		Mat diff_t = -0.5*diff;
		Mat b;
		exp(diff_t.t()*Spos.inv()*diff,b);
		
	    likelihood.at<double>(0,i) =(b.at<double>(0,0)*alpha*Pd/(m/V));
		sumLikelihood += likelihood.at<double>(0,i) ;
		
	}
	
	double normalizer = 1-Pd*Pg + sumLikelihood;
	double beta0 = (1-Pd*Pg)/normalizer;
	
	Mat betas = likelihood/normalizer;
	zHat = zHat.t();
	repeat(zHat,m,1,zHat);
	//zHat = zHat.t();
	Mat yi = measures_resize - zHat;

	repeat(betas, 1, 3,betas);

	Mat y;
	
	reduce(yi.mul(betas),y, 0,CV_REDUCE_SUM);
	y = y.t();
	Mat K = ((S.t()).inv()*(prediction.P*(H.t())).t()).t(); // kalman gain;
	
	Mat updatedX = prediction.x + K*y;
	
	Mat Pc = prediction.P - K*S*K.t();
	betas = betas.t();
	
	
	Mat Ptilde = K*(betas.mul(yi.t())*yi - y*y.t())*K.t();
	
	Mat updatedP = beta0*prediction.P +(1-beta0)*Pc + Ptilde;

	
	updatedP.copyTo(prediction.P);
	updatedX.copyTo(prediction.x);
}

vector<bool> CLegTracker::_not(vector<bool>ves)
{
	vector<bool>help;
	for(int i=0; i<ves.size(); i++)
	{
		help.push_back(1-ves[i]);
	}
	return help;
}

int CLegTracker::nnz(vector<bool>ves)
{
	int n = 0;
	for(int i=0; i<ves.size(); i++)
	{
		if(ves[i] == 1)
		{
			n++;
		}
	}

	return n;
}

Mat CLegTracker::delMatRow(Mat matrix, int row)
{
	Mat help = Mat::zeros(matrix.rows-1, matrix.cols, CV_64FC1);
	int idx, idy;
	for(int i=0; i<help.rows; i++)
	{

		for(int j=0; j<help.cols; j++)
		{
			if(i>=row)
			{ 
				idx= i+1;
			}
			else
			{
				idx = i;
			}
			
			help.ptr<double>(i)[j] = matrix.ptr<double>(idx)[j];

		}
	}
	return help;
}

Mat CLegTracker::delMat(Mat matrix, int row, int col)
{
	Mat help = Mat::zeros(matrix.rows-1, matrix.cols-1, CV_64FC1);
	int idx, idy;
	for(int i=0; i<help.rows; i++)
	{

		for(int j=0; j<help.cols; j++)
		{
			if(i>=row)
			{ 
				idx= i+1;
			}
			else
			{
				idx = i;
			}
			if(j>=col)
			{ 
				idy = j+1;
			}
			else
			{
				idy = j;
			}
			help.ptr<double>(i)[j] = matrix.ptr<double>(idx)[idy];
			
		}
	}
	return help;
}

Mat CLegTracker::delMatCol(Mat matrix, int col)
{
	Mat help = Mat::zeros(matrix.rows, matrix.cols-1, CV_64FC1);
	int  idy;
	for(int i=0; i<help.rows; i++)
	{

		for(int j=0; j<help.cols; j++)
		{
			if(j>=col)
			{ 
				idy= j+1;
			}
			else
			{
				idy = j;
			}
			
			help.ptr<double>(i)[j] = matrix.ptr<double>(i)[idy];

		}
	}
	return help;
}
template<typename T> bool  CLegTracker::ismember(T value,vector <T> vecs)
{
	for(int i=0; i<vecs.size(); i++)
	{
		if(value == vecs[i])
			return true;
	}
	return false;
}
void CLegTracker::pdist2(Point2d centroids1, vector<Point2d>centroids2, vector<double>&dist)
{
	for(int j=0; j<centroids2.size(); j++)
	{
		double dst = sqrt((centroids1.x-centroids2[j].x)*(centroids1.x-centroids2[j].x) + (centroids1.y-centroids2[j].y)* (centroids1.y-centroids2[j].y));
		dist.push_back(dst);
	}
	
}
void CLegTracker::pdist2(vector<Point2d>centroids1, vector<Point2d>centroids2, Mat &dist)
{
	dist = Mat::zeros(centroids1.size(), centroids2.size(), CV_64FC1);
	for(int i=0; i<centroids1.size(); i++)
	{
		
		for(int j=0; j<centroids2.size(); j++)
		{
			double dst = sqrt((centroids1[i].x-centroids2[j].x)*(centroids1[i].x-centroids2[j].x) + (centroids1[i].y-centroids2[j].y)* (centroids1[i].y-centroids2[j].y));
			dist.ptr<double>(i)[j] = dst;
		}
		
	}
}
void CLegTracker::_min(Mat dist, vector<double>&minVals, vector<int>&minIDs)
{
	for(int i=0; i<dist.rows; i++)
	{
		double minVal = FLT_MAX;
	    int minID = 0;
		for(int j=0; j<dist.cols; j++)
		{
			if(dist.ptr<double>(i)[j]<minVal)
			{
				minVal =dist.ptr<double>(i)[j]; 
				minID = j ;
			}
		}
		minVals.push_back(minVal);
		minIDs.push_back(minID);
	}
}
int CLegTracker:: _min(vector<double>_dist, double &minVal)
{
	int minID = -1;
	minVal = FLT_MAX;
	for(int i=0; i<_dist.size(); i++)
	{
		if(_dist[i]<minVal)
		{
			minVal = _dist[i];
			minID = i;
		}

	}
	return minID;
}
//void CLegTracker::setdiff(vector<int>A, vector<int>B, vector<int>C)
//{
//	set_difference(A.begin(),A.end(),B.begin(), B.end(),inserter(C,C.begin()));
//}
template<typename T> void CLegTracker::_find(vector<T>vecs, T val, int equal, vector<int>&res)
{
	if(equal == 0)
	{
		for(int i=0; i<vecs.size(); i++)
		{
			if (val == vecs[i])
			{
				res.push_back(i);
				
			}

		}
	
	}
	else if(equal == -1)
	{
		for(int i=0; i<vecs.size(); i++)
		{
			if (val > vecs[i])
			{
				res.push_back(i);
			}

		}
	
	}
	else if(equal == 1)
	{
		for(int i=0; i<vecs.size(); i++)
		{
			if (val < vecs[i])
			{
				res.push_back(i);
			}

		}
	}


}
template<typename T> int CLegTracker::_find(vector<T>vecs, T val)
{
	for(int i=0; i<vecs.size(); i++)
	{
		if(vecs[i] == val)
			return i;
	}
	return -1;

}
double CLegTracker::_minDist(Mat allPoints1, Mat allPoints2, double minThreshold)
{

	//parallelize computation of distances
	double *points1 =(double*) allPoints1.data;
	double *points2 =(double*) allPoints2.data;

	int numPoints1 = allPoints1.rows;
	int numPoints2 = allPoints2.rows;

	double maxDist = 5000*5000;

	double threshold = minThreshold*minThreshold;

	double minDist[MAX_THREAD];
	int id1[MAX_THREAD];
	int id2[MAX_THREAD];
	uint8_t numberOfThreads = 1;
	int stop = 0;
	for(int i=0; i<MAX_THREAD; i++)
	{
		minDist[i] = maxDist;
	}

#pragma omp parallel
	{
		int threadID   = omp_get_thread_num();
		int numThreads = omp_get_num_threads();
		
	
		numberOfThreads = numThreads;
		int pointsPerThread = numPoints1/numThreads;

		int start = threadID*pointsPerThread;
		int finish = start + pointsPerThread - 1;
		minDist[threadID] = maxDist;

		double x1,y1,z1,dist,dx,dy,dz;
		for (int i=start; i<=finish; i++) {
			x1 = points1[i*3];
			y1 = points1[i*3+1];
			z1 = points1[i*3+2];
			
			for (int j=0; j<numPoints2; j++) {
				
				dx = (x1-points2[j*3]);
				dy = (y1-points2[j*3+1]);
				dz = (z1-points2[j*3+2]);
				dist = dx*dx+dy*dy+dz*dz;
			

				if (dist < minDist[threadID]) {
					minDist[threadID] = dist;
					id1[threadID] = i;
					id2[threadID] = j;
				}
			}
			
			if (stop)
				break;
			
			if (minDist[threadID] < threshold) {
				stop = 1;
				break;
			}
		}
	}


	double minVal= maxDist;

	
	for (int k = 0; k<numberOfThreads; k++) {
		if (minDist[k] <minVal) {
			
			minVal = minDist[k];
	
		}
	}

	minVal= sqrt(minVal);
	return minVal;

}
void CLegTracker:: transferMatToVecVec(Mat matrix, vector<Point2d>&vec)
{
	double *ptrFootPoints = (double*)matrix.data;
	for(int i=0; i<matrix.rows; i++)
	{
		
		vec.push_back(Point2d( ptrFootPoints[3*i + 0], ptrFootPoints[3*i + 1]));
	
	}
}