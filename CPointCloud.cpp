#include "CPointCloud.h"

CPointCloud::CPointCloud()
{
	 pointsPtr = NULL;
	 colorsPtr = NULL;
	 rotatePointsPtr = NULL;
	 libSvm.libsvmreadmodel();
}


CPointCloud::~CPointCloud()
{

}


void CPointCloud::getVoxelGrid(Mat matDepth, Mat matImage)
{
	delete pointsPtr;
	delete colorsPtr;
	
	pointsPtr = NULL;
	colorsPtr = NULL;

	rows = matDepth.rows;
	cols = matDepth.cols;
	
	numPoints = rows*cols;
	uint16_t *depth = (uint16_t*) matDepth.data; // depth image ptr
	uint8_t  *rgb   = (uint8_t *) matImage.data;
	
	double *points = new double[3*numPoints];
	memset(points, 0, sizeof(double)*3*numPoints);

	uint8_t* colors = new uint8_t[3*numPoints];
	memset(colors, 0, sizeof(uint8_t)*3*numPoints);

	uint8_t*counts = new uint8_t[numPoints];
	memset(counts, 0, sizeof(uint8_t)*numPoints);

	uint16_t maxDepth = 3500; // mm
	double voxelDim = 20; // mm
	double sideLengthX = 6000/voxelDim; //in voxel
	double sideLengthY = 6000/voxelDim; //in voxel   

	register double fx, fy, cx, cy;
	if(rows == 240 ||cols == 320)
	{	//low resolution
		fx = 1/(525.0/2);
		fy = 1/(525.0/2);
		cx = 159.75;
		cy = 119.75;
	}

	else 
	{
		fx = 1/525.0;
		fy = 1/525.0;
		cx = 319.5;
		cy = 239.5;

	}
	
	//---------------------
	uint32_t offsets[MAX_THREAD];
	uint8_t numberOfThreads = 1;
	register double invVoxelDim = 1/voxelDim;

	#pragma omp parallel shared(numberOfThreads, depth, rgb, points,offsets, voxelDim, sideLengthX, sideLengthY)//,
	{
		int threadID        = omp_get_thread_num();
		int numThreads      = omp_get_num_threads();
		numberOfThreads     = numThreads;
		int pointsPerThread = numPoints/numThreads;
		uint32_t freeOffset = threadID*pointsPerThread;
		std::unordered_map<uint32_t, uint32_t> voxel;
		int start  = threadID * pointsPerThread;
		int finish = start    +  pointsPerThread - 1;

		double x, y, z;
		uint32_t offset, count;
		double invCount;
		for (int i=start; i<=finish; i++) 	
		{
			if (depth[i] <= 0 || depth[i] > maxDepth)
			{
				continue;
			}//if
			z = depth[i];
			x=  -((i%cols) - cx)*z*fx;
			y = -((i/cols) - cy)*z*fy; // assuming depth not mirrored
			
			uint32_t voxelID = (uint32_t)(x*invVoxelDim+sideLengthX/2) + (uint32_t)(y*invVoxelDim+sideLengthY/2)*sideLengthX + (uint32_t)(z*invVoxelDim)*sideLengthX*sideLengthY;    
			
			if (voxel.count(voxelID) == 0) 
			{
				voxel[voxelID]         = freeOffset;
				colors[3*freeOffset]   = rgb[3*i];//rgb[i];
				colors[3*freeOffset+1] = rgb[3*i+1];//rgb[i+numPoints];
				colors[3*freeOffset+2] = rgb[3*i+2];//rgb[i+2*numPoints];
				freeOffset++;
			}// if voxel

			offset = voxel[voxelID];
			count = (uint32_t)counts[offset];
			invCount = 1/(count+1.0);

			// average among the points falling in the same voxel
			points[3*offset]   = (points[3*offset]*count + x)   * invCount;
			points[3*offset+1] = (points[3*offset+1]*count + y) * invCount;
			points[3*offset+2] = (points[3*offset+2]*count + z) * invCount;

			counts[offset]++;

		}// for

		offsets[threadID] = freeOffset - threadID*pointsPerThread; 
	}

	int start = 0;
	for (int i=1; i<numberOfThreads; i++)
	{
		start += offsets[i-1];
		memcpy(&colors[3*start + 1], &colors[3*i*numPoints/numberOfThreads + 1], sizeof(uint8_t)*offsets[i]*3);
		memcpy(&points[3*start + 1], &points[3*i*numPoints/numberOfThreads + 1], sizeof(double)*offsets[i]*3); 
	}

	 num_after_voxel = start + offsets[numberOfThreads-1];
	
	 pointsPtr = new double [3*num_after_voxel];
	 colorsPtr = new uint8_t[3*num_after_voxel];

#pragma omp parallel for
	 for(int i=0; i<num_after_voxel; i++)
	 {
		 pointsPtr[3*i]   = points[3*i];
		 pointsPtr[3*i+1] = points[3*i +1];
		 pointsPtr[3*i+2] = points[3*i+2];
	
		 colorsPtr[3*i]    = colors[3*i];
		 colorsPtr[3* i+1] = colors[3*i+1];
	     colorsPtr[3* i+2] = colors[3*i +2];
	 }
	
	delete points;
	delete colors;
	delete counts;

}

bool CPointCloud::getGroundPlane(PlaneParmeters& best_plane, float maxInclinationAngle , float tol )
{
	float collinearityThreshold = sin(0.1*CV_PI/180);
	if(maxInclinationAngle >90)
	{
		maxInclinationAngle = 90;
	}
	if(maxInclinationAngle <10)
	{
		maxInclinationAngle = 10;
	}
	tol = 2;	//default 1e-6
	float minTolWall = deg2rad(maxInclinationAngle);
	float maxTolWall = deg2rad(180 - maxInclinationAngle);

	int tentative_num = 30;
	vector<Point3d>ptsCut;
	//cut points;
	for(int i=0; i<num_after_voxel; i++)
	{
		if(pointsPtr[3*i+1]<0)
		{
			ptsCut.push_back(Point3d(pointsPtr[3*i], pointsPtr[3*i+1], pointsPtr[3*i+2]));
		}
	}

	int cutSize = ptsCut.size();
	if(cutSize< tentative_num*3)
	{
		best_plane.a = 0;
		best_plane.b = 0;
		best_plane.c = 0;
		best_plane.d = 0;
		cout<< "no enough poitns to estimate ground plane!"<<endl;
		return false;
	}

	int best_count = 0;
	srand((int)time(0));
	while(tentative_num>0)
	{
		
		int id1 = rand()%cutSize;
		int id2 = rand()%cutSize;
		int id3 = rand()%cutSize;

		Point3d P1,P2,P3;
		P1 = ptsCut[id1];  
		P2 = ptsCut[id2];
		P3 = ptsCut[id3];
		
		Point3d u, v;
		u = P2 - P1;
		
		double normU = sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
		u.x = u.x/normU;
		u.y = u.y/normU;
		u.z = u.z/normU;

		v = P3- P1;
		double normV = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
		v.x = v.x/normU;
		v.y = v.y/normU;
		v.z = v.z/normU;

		// plane normal
		double planeNormalX = u.y*v.z - u.z*v.y; 
		double planeNormalY = u.z*v.x - u.x*v.z; 
		double planeNormalZ = u.x*v.y - u.y*v.x;

		double planeNorm = sqrt((double)(planeNormalX*planeNormalX +planeNormalY*planeNormalY +planeNormalZ*planeNormalZ));
		if(planeNorm < collinearityThreshold)
		{
			
			continue;
		}
		double planeToCheckX = planeNormalX/planeNorm;
		double planeToCheckY = planeNormalY/planeNorm;
		double planeToCheckZ = planeNormalZ/planeNorm;

		double angleWithYaxis = acos(planeToCheckY);

		if(angleWithYaxis>minTolWall &&angleWithYaxis <maxTolWall)
		{
			continue;
		}

		double d = (-planeToCheckX*P1.x) + (-planeToCheckY*P1.y) + (-planeToCheckZ*P1.z);

		PlaneParmeters plane;
		plane.a = planeToCheckX/d;
		plane.b = planeToCheckY/d;
		plane.c = planeToCheckZ/d;
		plane.d = d/d;

		int count = nnz(pointsPtr, plane, tol);
	
		if(count>best_count)
		{
			best_count = count;
			best_plane = plane;
		}

		tentative_num = tentative_num - 1;

	} //while
	if(best_plane.b <0)
	{
		best_plane =best_plane*(-1);	
	}
	return true;
}

void CPointCloud::rotatePointCloud(PlaneParmeters floorParmeters, Mat& matInv)
{
	PlaneParmeters floor;
	if(floorParmeters.b <0 )
	{
		floor = floorParmeters*(-1);
	}
	else
	{
		floor = floorParmeters;
	}
	
	float normFloor = sqrt(floor.a*floor.a + floor.b*floor.b + floor.c*floor.c);
	
	float floorVersorX = floor.a/normFloor;
	float floorVersorY = floor.b/normFloor;
	float floorVersonZ = floor.c/normFloor;

	float tol = 1e-6;
	float sum = floorVersorX*floorVersorX + floorVersorY*floorVersorY + (floorVersonZ-1)*(floorVersonZ-1);
	bool isDo = sqrt(sum)<tol;
	if(isDo)
	{
		rotatePointsPtr = pointsPtr; 
		return;			 
	}
	rotatePointsPtr = new double[num_after_voxel*3];
	

	// obtain the axis (see angle next) of rotation of camera zAxis w.r.t. the floorVersor
	float axisOfRotationX = floorVersorY;
	float axisOfRotationY = -floorVersorX;
	float axisOfRotationZ = 0;
	float normAxisOfRotation = sqrt(axisOfRotationX*axisOfRotationX + axisOfRotationY*axisOfRotationY + axisOfRotationZ*axisOfRotationZ);

	axisOfRotationX = axisOfRotationX/normAxisOfRotation;
	axisOfRotationY = axisOfRotationY/normAxisOfRotation;
	axisOfRotationZ = axisOfRotationZ/normAxisOfRotation;

	// 1) determine rotation of the x-axis to correct eventual roll of camera   
	// roll = acos(dot(axisOfRotation, xAxis)/(norm(axisOfRotation)*norm(xAxis)));
	float normAxisOfRotation2 = sqrt((float)(axisOfRotationX*axisOfRotationX + axisOfRotationY*axisOfRotationY + axisOfRotationZ*axisOfRotationZ));
	float roll = acos(axisOfRotationX/normAxisOfRotation2);

	float angleRotX;
	if(axisOfRotationY >0)
	{
		angleRotX = -roll;
	}
	else
	{
		angleRotX = +roll;
	}

	Mat RotOfX, RotOfXY, R, rot;
	rotationMatrixFromAxisAndAngle(0, 0, 1, angleRotX, RotOfX);
 
	// 2) rotate camera plane xy of CV_PI
	rotationMatrixFromAxisAndAngle(0, 0, 1,-CV_PI, RotOfXY);
	
	// 3) rotate around new x-axis to make zAxis pointing to plane versor direction
	float floorVersorNorm = sqrt(floorVersorX*floorVersorX + floorVersorY*floorVersorY+floorVersonZ*floorVersonZ);

	// angle between floor Versor and zAxis
	float angle = acos(floorVersonZ/floorVersorNorm) ;
	
	rotationMatrixFromAxisAndAngle(1, 0, 0, -angle, R);

	rot = R * RotOfXY *RotOfX;
	
	float dist = abs(floor.d)/normFloor;

	for(int i=0; i<num_after_voxel; i++)
	{
		 rotatePointsPtr[3*i]   = rot.at<double>(0,0)*pointsPtr[3*i] +rot.at<double>(0,1)*pointsPtr[3*i+1] + rot.at<double>(0,2)*pointsPtr[3*i+2] + 0;
		 rotatePointsPtr[3*i+1] = rot.at<double>(1,0)*pointsPtr[3*i] +rot.at<double>(1,1)*pointsPtr[3*i+1] + rot.at<double>(1,2)*pointsPtr[3*i+2] + 0;
		 rotatePointsPtr[3*i+2] = rot.at<double>(2,0)*pointsPtr[3*i] +rot.at<double>(2,1)*pointsPtr[3*i+1] + rot.at<double>(2,2)*pointsPtr[3*i+2] + dist;

	}
	Mat invRotOfX ,invRotOfXY, invR;
	rotationMatrixFromAxisAndAngle(0, 0, 1,-angleRotX, invRotOfX);
	rotationMatrixFromAxisAndAngle(0, 0, 1, CV_PI, invRotOfXY);
	rotationMatrixFromAxisAndAngle(1, 0, 0, angle, invR);

	Mat invRot = invRotOfX *invRotOfXY *invR;
	double x1 = invRot.at<double>(0,0);
	double y1 = invRot.at<double>(0,1);
	double z1 = invRot.at<double>(0,2);

	double x2 = invRot.at<double>(1,0);
	double y2 = invRot.at<double>(1,1);
	double z2 = invRot.at<double>(1,2);

	double x3 = invRot.at<double>(2,0);
	double y3 = invRot.at<double>(2,1);
	double z3 = invRot.at<double>(2,2);

	matInv = (Mat_<double>(4,4)<<x1, y1, z1, z1*(-dist),
								 x2, y2, z2, z2*(-dist),
								 x3, y3, z3, z3*(-dist),
								 0,  0,  0,  1);
	
	matInv.copyTo(invM);
}

void CPointCloud::getCandidateLegs(vector<Candidate>&cadidates)
{

	double footCutHeight = 210;   //mm
	double personCutHeight = 2000; //mm
	double footCutBase = 30;
	double legCutBase = 10;

	vector<Point3d> voxelGridPoints;
	vector<Point3d> footCutPoints;

	vector<Point3i> voxelGridColors;
	vector<Point3i> footCutColors;

	for(int i=0; i<num_after_voxel; i++)
	{
		double height = rotatePointsPtr[3*i+2];
		if(height >legCutBase && height<personCutHeight)
		{	
			if(height>footCutBase && height<footCutHeight) // get footCutPoints
			{
				footCutPoints.push_back(Point3d(rotatePointsPtr[3*i+0], rotatePointsPtr[3*i+1], rotatePointsPtr[3*i+2]));
				footCutColors.push_back(Point3d(colorsPtr[3*i],colorsPtr[3*i+1],colorsPtr[3*i+2]));
			}
			else
			{
				voxelGridPoints.push_back(Point3d(rotatePointsPtr[3*i+0], rotatePointsPtr[3*i+1], rotatePointsPtr[3*i+2]));
				voxelGridColors.push_back(Point3d(colorsPtr[3*i+0],colorsPtr[3*i+1],colorsPtr[3*i+2]));
			}
		}
	}
	
	if(footCutPoints.size() == 0)
	{
		return;
	}
	vector<int>classes(footCutPoints.size());
	vector<int>types(footCutPoints.size());

	dbscanClustering(Mat(footCutPoints), 5, 40,classes, types);

	int nClusters;
	vector<int>::iterator biggest = max_element( begin(classes), end(classes));
	nClusters = *biggest;
	
	double minTol = 50; // min foot size in mm
	double maxTol = 600;// max foot size in mm
	double minNumPoints = 20; //minimum number of points to consider a cluster

	vector<vector<Point3d>>nFootPoints(nClusters);
	vector<vector<Point3d>>nFootColors(nClusters);

	vector<int>clusterOK(nClusters);
	vector<int>candidateOK(nClusters);

	Mat centroidsOnFloorTmp = Mat::zeros(Size(2,nClusters),CV_64FC1);

	for(int i=0; i<footCutPoints.size(); i++)
	{
		int id = classes[i];
		if(id<=0) continue;
		nFootPoints[id-1].push_back(footCutPoints[i]);
		nFootColors[id-1].push_back(footCutColors[i]);
	}
    
	for(int i=0; i<nClusters; i++)
	{
		Candidate cadTmp;
		if(nFootPoints[i].size()<minNumPoints)
		{
			clusterOK[i] = 0;
			continue;
		}
	    minBoundingBox(nFootPoints[i],cadTmp.rectangleOnFloor);

		double x0 = cadTmp.rectangleOnFloor[0].x;
		double y0 = cadTmp.rectangleOnFloor[0].y;

		double x1 = cadTmp.rectangleOnFloor[1].x;
		double y1 = cadTmp.rectangleOnFloor[1].y;

		double x2 = cadTmp.rectangleOnFloor[2].x;
		double y2 = cadTmp.rectangleOnFloor[2].y;

		double l1 = sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1));
		double l2 = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
	
		if(l1<minTol && l2<minTol)
		{
			clusterOK[i] = 0;
			continue;
		}

		clusterOK[i] = 1;

		if (l1<maxTol && l2<maxTol && l1>minTol && l2>minTol)
		{
			candidateOK[i] = 1;
			float sumX = 0;
			float sumY = 0;
			for(int m=0; m<nFootPoints[i].size(); m++)
			{
				sumX += nFootPoints[i][m].x;
				sumY += nFootPoints[i][m].y;
			}
			centroidsOnFloorTmp.at<double>(i,0) =(double) sumX/nFootPoints[i].size();
			centroidsOnFloorTmp.at<double>(i,1) = (double)sumY/nFootPoints[i].size();

		
			cadTmp.footPoints = Mat::zeros(nFootPoints[i].size(),1,CV_64FC3);
			cadTmp.footPoints = Mat::zeros(nFootColors[i].size(),1,CV_8UC3);
			Mat(nFootColors[i]).copyTo(cadTmp.footColors);
			Mat(nFootPoints[i]).copyTo(cadTmp.footPoints);
			//cadTmp.footPoints = Mat(nFootPoints[i]);
			cadTmp.centroid = Point2d(centroidsOnFloorTmp.at<double>(i,0),centroidsOnFloorTmp.at<double>(i,1));
			cadidates.push_back(cadTmp); 
		}    
		
  	} //for
	Mat centroidsOnFloor = Mat::zeros(Size(2,nClusters),CV_64FC1);
	int forward= 0;
	int backward = centroidsOnFloorTmp.rows -1;
	for(int  i=0; i<centroidsOnFloorTmp.rows; i++)
	{
		if(candidateOK[i] == 0)
		{
			centroidsOnFloor.at<double>(backward, 0) = centroidsOnFloorTmp.at<double>(i,0);
			centroidsOnFloor.at<double>(backward, 1) = centroidsOnFloorTmp.at<double>(i,1);
			backward--;
		}
		else
		{
		  centroidsOnFloor.at<double>(forward, 0) = centroidsOnFloorTmp.at<double>(i,0);
		  centroidsOnFloor.at<double>(forward, 1) = centroidsOnFloorTmp.at<double>(i,1);
		  forward++;
		}
		
	}
	expandCandidates(cadidates, centroidsOnFloor,Mat(voxelGridPoints), Mat(voxelGridColors));
}



void CPointCloud::expandCandidates(vector<Candidate>&candidates, Mat centroidOnFloorM, Mat voxelGridPoints, Mat voxelGridColors)
{
	uint8_t *colors = (uint8_t*)voxelGridColors.data;

	int srow = SHEIGHT/SSIDE;
	int scol = SWIDTH/SSIDE;
	
	//build kd-tree of all point cloud
	PointCloud pointCloud;
	
	pointCloud.pts = (double*)voxelGridPoints.data;
	pointCloud.numPoints = voxelGridPoints.rows;
	KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud>, PointCloud, 3 /* dim */> tree(3 /*dim*/, pointCloud, KDTreeSingleIndexAdaptorParams(20 /* max leaf */) );
	tree.buildIndex();
	
	size_t numCandidates = candidates.size();
	size_t numPoints = voxelGridPoints.rows;

	double* centroidsOnFloor = (double*)centroidOnFloorM.data;
	size_t  numCentroidsOnFloor = centroidOnFloorM.rows;

	//allocations for filling candidates structs
	double **footPointsPtrs = (double **)malloc(numCandidates*sizeof(double *));
	double **legPointsPtrs = (double **)malloc(numCandidates*sizeof(double *));
	double **legPointsNormalizedPtrs = (double **)malloc(numCandidates*sizeof(double *));
	double **silhouettePtrs = (double **)malloc(numCandidates*sizeof(double *));
	double **personPointsPtrs = (double **)malloc(numCandidates*sizeof(double *));
	size_t *numFootPointsPtrs = (size_t *)malloc(numCandidates*sizeof(size_t));
	size_t *actualNumLegPointsPtrs = (size_t *)malloc(numCandidates*sizeof(size_t));
	size_t *actualNumPersonPointsPtrs = (size_t *)malloc(numCandidates*sizeof(size_t));
	uint8_t **legColorsPtrs = (uint8_t **)malloc(numCandidates*sizeof(uint8_t *));
	uint8_t **footColorsPtrs = (uint8_t **)malloc(numCandidates*sizeof(uint8_t *));
	
	for (int i=0; i < numCandidates; i++)
	{
		footPointsPtrs[i]   = (double*)candidates[i].footPoints.data;
		numFootPointsPtrs[i] = candidates[i].footPoints.rows;
		// legPoints
		candidates[i].legPoints = Mat::zeros(Size(1,numPoints+numFootPointsPtrs[i]),CV_64FC3);
		legPointsPtrs[i]    = (double*)candidates[i].legPoints.data;
		//legPointsNormalized
		candidates[i].legPointsNormalized = Mat::zeros(Size(1, numPoints+numFootPointsPtrs[i]), CV_64FC3);
		legPointsNormalizedPtrs[i] = (double*)candidates[i].legPointsNormalized.data;
		//silhouette
		candidates[i].silhouette = Mat::zeros(srow, scol, CV_64FC1);
		silhouettePtrs[i]   = (double*)candidates[i].silhouette.data;
		// allPoints
		candidates[i].allPoints = Mat::zeros(Size(1, numPoints+numFootPointsPtrs[i]), CV_64FC3);
		personPointsPtrs[i] = (double*)candidates[i].allPoints.data;
		//footColors
		footColorsPtrs[i]   = (uint8_t*)candidates[i].footColors.data;
		// legColors
		candidates[i].legColors = Mat::zeros(Size(1,numPoints+numFootPointsPtrs[i]),CV_8UC3);
		legColorsPtrs[i]    = (uint8_t*)candidates[i].legColors.data;
	}// for

	// array for points already considered for explansion
	uint8_t *visited = (uint8_t*)calloc(numPoints, sizeof(uint8_t));

	//expand candidates in parallel
	int freeCandidate = 0;
	#pragma omp parallel
	{
	   int id;
	   while(true)
	   {
			#pragma omp critical
		    {
				id = freeCandidate;
				freeCandidate++;
	    	}
			if(id >= numCandidates)
				break;
			expandCandidate(footPointsPtrs[id], 
							legPointsPtrs[id],
							personPointsPtrs[id], 
							centroidsOnFloor, 
							&tree, 
							numFootPointsPtrs[id],
							&actualNumLegPointsPtrs[id],
							&actualNumPersonPointsPtrs[id], 
							numCentroidsOnFloor,
							visited,
							pointCloud.pts, 
							id,
							colors,
							legColorsPtrs[id],
							footColorsPtrs[id], 
							legPointsNormalizedPtrs[id], 
							silhouettePtrs[id], numPoints);
	   }
	 
	}
	//
	for(int i=0; i<candidates.size(); i++)
	{
		// set legPoints actual num legs
		candidates[i].legPoints = candidates[i].legPoints.rowRange(0, actualNumLegPointsPtrs[i]);
		candidates[i].legColors = candidates[i].legColors.rowRange(0, actualNumLegPointsPtrs[i]);
		candidates[i].legPointsNormalized = candidates[i].legPointsNormalized.rowRange(0, actualNumLegPointsPtrs[i]);
	    candidates[i].allPoints = candidates[i].allPoints.rowRange(0, actualNumPersonPointsPtrs[i]);
	}
	free(footPointsPtrs);
	free(legPointsPtrs);
	free(legPointsNormalizedPtrs);
	free(personPointsPtrs);
	free(numFootPointsPtrs);
	free(actualNumPersonPointsPtrs);
	free(actualNumLegPointsPtrs);
	free(silhouettePtrs);
	free(visited);
	free(footColorsPtrs);
	free(legColorsPtrs);
}

void CPointCloud::expandCandidate(double *footPoints,           //footPoints of a candidates leg              
								  double *legPoints, 
								  double *personPoints, 
								  double *centroidsOnFloor, 
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
								  int numPoints) 

{
	
	int legCutHeight = LEG_CUT_HEIGHT;

	double eps = DBSCAN_RADIUS;
	int k = DBSCAN_NEIGH;
	int reserveSize = 50;

	double xMin=4000, yMin=4000, xMax = 0;

	// foot centroid
	double xCenter = centroidsOnFloor[idCandidate*2];
	double yCenter = centroidsOnFloor[idCandidate*2+1];

	// foot angle on the scene
	double alpha = acos(yCenter/sqrt(xCenter*xCenter + yCenter*yCenter));

	if (xCenter<0) 
		alpha = -alpha;
	double cosAlpha = cos(alpha);
	double sinAlpha = sin(alpha);

	bool enabledColor = true;
	if (colors == NULL || legColors == NULL || footColors == NULL)
		enabledColor = false;

	// copy foot data in leg fields
	memcpy(legPoints, footPoints, numFootPoints*sizeof(double)*3);
	if (enabledColor)
		memcpy(legColors, footColors, numFootPoints*sizeof(uint8_t)*3);

	*actualNumLegPoints = numFootPoints;

	const double search_radius = static_cast<double>(eps*eps);
	nanoflann::SearchParams params;
	params.sorted = false;

	queue<size_t> neigh;

	double queryPoint[3];

	std::vector<std::pair<size_t, double> >  ret_matches;

	// find immediate neighbors of foot points, from which it will start the expansion
	for (int i=0; i<numFootPoints; i++) {
		queryPoint[0] = footPoints[i*3];
		queryPoint[1] = footPoints[i*3+1];
		queryPoint[2] = footPoints[i*3+2];

		legPointsNormalized[3*i] = cosAlpha*queryPoint[0]-sinAlpha*queryPoint[1];

		legPointsNormalized[3*i+1] = sinAlpha*queryPoint[0]+cosAlpha*queryPoint[1];
		legPointsNormalized[3*i+2] = queryPoint[2];

		// find min and max coordinates for normalize the candidate
		if (legPointsNormalized[3*i] < xMin)
				xMin = legPointsNormalized[3*i];

		if (legPointsNormalized[3*i] > xMax)
				xMax = legPointsNormalized[3*i];

		if (legPointsNormalized[3*i+1] < yMin)
				yMin = legPointsNormalized[3*i+1];

		// search neighbors of the current foot point
		ret_matches.clear();
		ret_matches.reserve(reserveSize);
		const size_t nMatches = (*tree).radiusSearch(queryPoint, search_radius, ret_matches, params);

		if (nMatches >= k) {
			// border connected with outer points
			for(int j=0;j<nMatches;j++) {
				size_t id = ret_matches[j].first;
				if(visited[id] == 0) {
					// if not already visited add to expansion queue
					neigh.push(id);
					visited[id] = 1;
					legPoints[(*actualNumLegPoints)*3]   = points[3*id];
					legPoints[(*actualNumLegPoints)*3+1] = points[3*id+1]; 
					legPoints[(*actualNumLegPoints)*3+2] = points[3*id+2];

					legPointsNormalized[(*actualNumLegPoints)*3]   = cosAlpha*points[3*id]-sinAlpha*points[3*id+1];
					legPointsNormalized[(*actualNumLegPoints)*3+1] = sinAlpha*points[3*id]+cosAlpha*points[3*id+1];
					legPointsNormalized[(*actualNumLegPoints)*3+2] = points[3*id+2];

					if (legPointsNormalized[(*actualNumLegPoints)*3] < xMin)
					{
							xMin = legPointsNormalized[(*actualNumLegPoints)*3];
					}

					if (legPointsNormalized[(*actualNumLegPoints)*3] > xMax)
					{
							xMax = legPointsNormalized[(*actualNumLegPoints)*3];
					}

					if (legPointsNormalized[(*actualNumLegPoints)*3+1] < yMin)
					{
							yMin = legPointsNormalized[(*actualNumLegPoints)*3+1];
					}

					if (enabledColor) 
					{
						legColors[(*actualNumLegPoints)*3]   = colors[3*id];
						legColors[(*actualNumLegPoints)*3+1] = colors[3*id+1]; 
						legColors[(*actualNumLegPoints)*3+2] = colors[3*id+2];
					}

					*actualNumLegPoints += 1;
				}
			}
		}
	}


	// copy leg points into person points
	memcpy(personPoints, legPoints, (*actualNumLegPoints)*sizeof(double)*3);
	*actualNumPersonPoints = *actualNumLegPoints;

	double x,y,z;

	// expand the leg and the person starting from the just found border points
	while(!neigh.empty()) {
		size_t id = neigh.front();
		neigh.pop();

		queryPoint[0] = points[id*3];
		queryPoint[1] = points[id*3+1];
		queryPoint[2] = points[id*3+2];

		// neighbors search
		ret_matches.clear();
		ret_matches.reserve(reserveSize);
		const size_t nMatches = (*tree).radiusSearch(queryPoint, search_radius, ret_matches, params);

		if (nMatches >= k+1) {
			// core point
			for(int j=0;j<nMatches;j++) 
			{
				size_t idx = ret_matches[j].first;
				if(visited[idx] == 0)
				{
					// check if belong to the current voronoi region
					double minDist = 4000*4000;
					int label = -1;
					x = points[idx*3];
					y = points[idx*3+1];
					z = points[idx*3+2];
					for (int t=0; t<numCentroidsOnFloor; t++) 
					{
						// Voronoi
						double dist = (x-centroidsOnFloor[t*2])*(x-centroidsOnFloor[t*2]) + (y-centroidsOnFloor[t*2+1])*(y-centroidsOnFloor[t*2+1]);
						if (dist<minDist) {
							minDist = dist;
							label = t;
						}
					}
					if (label == idCandidate)
					{
						// add to current expansion
						neigh.push(idx);
						visited[idx] = 1;
						if (*actualNumPersonPoints >= numPoints+numFootPoints) {
							//mexPrintf("numFootPoints: %d numPoints: %d actualPerson: %d actualLeg: %d\n", numFootPoints, numPoints, *actualNumPersonPoints, *actualNumLegPoints);
							continue;
						}

						// add to person points
						personPoints[(*actualNumPersonPoints)*3] = x;
						personPoints[(*actualNumPersonPoints)*3+1] = y; 
						personPoints[(*actualNumPersonPoints)*3+2] = z;

						*actualNumPersonPoints += 1;

						if (z < legCutHeight)
						{
							// add to leg points
							legPoints[(*actualNumLegPoints)*3] = x;
							legPoints[(*actualNumLegPoints)*3+1] = y; 
							legPoints[(*actualNumLegPoints)*3+2] = z;

							legPointsNormalized[(*actualNumLegPoints)*3] = cosAlpha*x-sinAlpha*y;
							legPointsNormalized[(*actualNumLegPoints)*3+1] = sinAlpha*x+cosAlpha*y;
							legPointsNormalized[(*actualNumLegPoints)*3+2] = z;

							if (legPointsNormalized[(*actualNumLegPoints)*3] < xMin)
								xMin = legPointsNormalized[(*actualNumLegPoints)*3];
							if (legPointsNormalized[(*actualNumLegPoints)*3] > xMax)
								xMax = legPointsNormalized[(*actualNumLegPoints)*3];
							if (legPointsNormalized[(*actualNumLegPoints)*3+1] < yMin)
								yMin = legPointsNormalized[(*actualNumLegPoints)*3+1];


							if (enabledColor) 
							{
								legColors[(*actualNumLegPoints)*3] = colors[idx*3];
								legColors[(*actualNumLegPoints)*3+1] = colors[idx*3+1]; 
								legColors[(*actualNumLegPoints)*3+2] = colors[idx*3+2];
							}

							*actualNumLegPoints += 1;
						}
					}
				}
			}
		}
	}
	
	// normalize legPoints and create silhouette
	int srow = SHEIGHT/SSIDE;
	int scol = SWIDTH/SSIDE;
	int sdepth = SDEPTH/SSIDE;
	double xoff = (SWIDTH-(xMax-xMin))/2;
	uint8_t *depthIndex = (uint8_t *)malloc(srow*scol*sizeof(uint8_t));
	uint8_t *count = (uint8_t *)calloc(srow*scol,sizeof(uint8_t));
	int currentRow, currentCol, currentDepth, linearIdx;

	// compute linear idx where each leg points belong
	for (int i=0; i<*actualNumLegPoints; i++)
	{
		legPointsNormalized[3*i] -= xMin;
		legPointsNormalized[3*i+1] -= yMin;

		currentRow = srow-1 - (int)(legPointsNormalized[3*i+2]/SSIDE);
		currentCol = (int)((legPointsNormalized[3*i]+xoff)/SSIDE);
		currentDepth = (int)(legPointsNormalized[3*i+1]/SSIDE)+1;
		if (currentRow<0 || currentCol<0 || currentDepth<0 || currentRow>srow-1 || currentCol>scol-1 || currentDepth>sdepth-1) continue;

		linearIdx = currentCol*srow + currentRow;
		if (count[linearIdx] == 0 || currentDepth < depthIndex[linearIdx])
		{
			count[linearIdx] = 1;
			silhouette[linearIdx] = legPointsNormalized[3*i+1];
			depthIndex[linearIdx] = currentDepth;
		}
		else if (currentDepth == depthIndex[linearIdx]) 
		{
			silhouette[linearIdx] = (silhouette[linearIdx]*count[linearIdx] + legPointsNormalized[3*i+1])/(count[linearIdx] + 1);
			count[linearIdx]++;
		}
	}

	// fill silhouette holes
	uint8_t *binary = depthIndex;
	memset(binary, 0, srow*scol*sizeof(uint8_t));
	for (int i=0; i<srow*scol; i++)
	{
		if (count[i] == 0)
		{
			silhouette[i] = SDEPTH;
		} else 
		{   
			binary[i] = 2;
		}
	}
	
	// dilate
	for (int i=1; i<scol-1; i++) 
	{
		for (int j=1; j<srow-1; j++)
		{
			if (binary[i*srow+j] == 2) 
			{
				if (binary[(i-1)*srow+j] == 0) binary[(i-1)*srow+j] = 3;
				if (binary[(i-1)*srow+j-1] == 0) binary[(i-1)*srow+j-1] = 3;
				if (binary[(i)*srow+j-1] == 0) binary[(i)*srow+j-1] = 3;
				if (binary[(i+1)*srow+j-1] == 0) binary[(i+1)*srow+j-1] = 3;
				if (binary[(i+1)*srow+j] == 0) binary[(i+1)*srow+j] = 3;
				if (binary[(i+1)*srow+j+1] == 0) binary[(i+1)*srow+j+1] = 3;
				if (binary[(i)*srow+j+1] == 0) binary[(i)*srow+j+1] = 3;
				if (binary[(i-1)*srow+j+1] == 0) binary[(i-1)*srow+j+1] = 3;
			}
		}
	}
	// erode
	for (int i=1; i<scol-1; i++) 
	{
		for (int j=1; j<srow-1; j++)
		{
			if (binary[i*srow+j] >= 2) 
			{
				if (binary[(i-1)*srow+j] == 0)
				{ 
					binary[(i)*srow+j] = 1;
					continue;
				}
				if (binary[(i)*srow+j-1] == 0) 
				{ 
					binary[(i)*srow+j] = 1;
					continue;
				}

				if (binary[(i+1)*srow+j-1] == 0) 
				{ 
					binary[(i)*srow+j] = 1;
					continue;
				}
				if (binary[(i+1)*srow+j] == 0) 
				{ 
					binary[(i)*srow+j] = 1;
					continue;
				}
				if (binary[(i+1)*srow+j+1] == 0) 
				{ 
					binary[(i)*srow+j] = 1;
					continue;
				}
				if (binary[(i)*srow+j+1] == 0)
				{ 
					binary[(i)*srow+j] = 1;
					continue;
				}
				if (binary[(i-1)*srow+j+1] == 0)
				{ 
					binary[(i)*srow+j] = 1;
					continue;
				}
				if (binary[(i-1)*srow+j-1] == 0) 
				{ 
					binary[(i)*srow+j] = 1;
					continue;
				}
			}
		}
	}
	// fill hole in correspondence of 3 in binary image with average of non zero neighbors
	double *silhouetteCopy = (double *)malloc(srow*scol*sizeof(double));
	silhouetteCopy = (double *)memcpy(silhouetteCopy, silhouette, srow*scol*sizeof(double));
	int numNeigh;
	double mean;
	for (int i=1; i<scol-1; i++) {
		for (int j=1; j<srow-1; j++) {
			if (binary[i*srow+j] == 3) {
				mean = 0;
				numNeigh = 0;
				if (binary[(i-1)*srow+j] == 2) { 
					mean += silhouetteCopy[(i-1)*srow+j];
					numNeigh++;
				}
				if (binary[(i)*srow+j-1] == 2) { 
					mean += silhouetteCopy[(i)*srow+j-1];
					numNeigh++;
				}
				if (binary[(i+1)*srow+j-1] == 2) { 
					mean += silhouetteCopy[(i+1)*srow+j-1];
					numNeigh++;
				}
				if (binary[(i+1)*srow+j] == 2) { 
					mean += silhouetteCopy[(i+1)*srow+j];
					numNeigh++;
				}
				if (binary[(i+1)*srow+j+1] == 2) { 
					mean += silhouetteCopy[(i+1)*srow+j+1];
					numNeigh++;
				}
				if (binary[(i)*srow+j+1] == 2) { 
					mean += silhouetteCopy[(i)*srow+j+1];
					numNeigh++;
				}
				if (binary[(i-1)*srow+j+1] == 2) { 
					mean += silhouetteCopy[(i-1)*srow+j+1];
					numNeigh++;
				}
				if (binary[(i-1)*srow+j-1] == 2) { 
					mean += silhouetteCopy[(i-1)*srow+j-1];
					numNeigh++;
				}

				silhouette[i*srow+j] = mean/numNeigh;

			}
		}
	}

	free(depthIndex);
	free(count);
	free(silhouetteCopy);
}

void CPointCloud::predictClass(vector<Candidate>cadidates, Mat &prob)
{
	if(cadidates.size() == 0) 
		return;
	Mat features;
	extractFeaturesHOG(cadidates, features);
	libSvm.libpreidctfast(features ,prob);
	
	//////// field of view on the map
	//float yMax = 3.5;
	//float fieldOfView =(float)57/180*CV_PI;
	//float xMin = -tan(fieldOfView/2)*yMax;
	//float xMax = tan(fieldOfView/2)*yMax;
	//float yMin = 0;

	//Mat img = Mat::zeros(Size(640,480),CV_8UC3);
	//float xOriMin = int((-tan(fieldOfView/2)*0.5)*320/2.5)+320;
	//float xOriMax =  int((tan(fieldOfView/2)*0.5)*320/2.5)+320;


	//line(img, Point(xOriMin,480),Point(xMin*320/2.5+320,0),Scalar::all(255),1,8,0);
	//line(img, Point(xOriMax,480),Point(xMax*320/2.5+320,0),Scalar::all(255),1,8,0);
	//
	//for(int i=0; i<cadidates.size(); i++)
	//{
	//	Point center = cadidates[i].centroid;

	//	center.x = (center.x*0.001)*320/2.5+320;
	//	center.y = 480-(center.y*0.001)*240/3;
	//	
	//	//center.x = max(min(center.x, 640),0);
	//	//center.y = max(min(center.y, 480),0);
	//	float minBorder = (-tan(fieldOfView/2)*cadidates[i].centroid.y*0.001)*320/2.5+320;
	//	float maxBorder = (tan(fieldOfView/2)*cadidates[i].centroid.y*0.001)*320/2.5+320;
	//	if(prob.at<double>(1, i)>0.80 )
	//	//cout<<minBorder<<"-"<<center.x<<"-"<<maxBorder<<endl;
	//   //if(center.x>maxBorder ||center.x<minBorder )
	//	//	continue;
	//	
	//	circle(img, center,4,Scalar(0,0,255),-1,8,0);
	//}

	//imshow("detect",img);
	
}

void CPointCloud::extractFeaturesHOG(vector<Candidate>candidates, Mat& features)
{
	size_t numCandidates = candidates.size();
	
	double **silhouettePtrs = (double **)malloc(numCandidates*sizeof(double *));
	
	Mat firstSilhouette = candidates[0].silhouette;
	
	int srow = firstSilhouette.rows;
	int scol = firstSilhouette.cols;

	silhouettePtrs[0] = (double*)firstSilhouette.data;

	// number of vertical and horizontal cells
	int nPatchV = srow/V_PATCH_SIZE;
	int nPatchH = scol/H_PATCH_SIZE;

	features = Mat::zeros(nPatchV*nPatchH*NBINS, numCandidates , CV_64FC1);

	double *featuresPtr =(double*) features.data;
	for(int i=1; i<numCandidates; i++)
	{
		silhouettePtrs[i] =(double*) candidates[i].silhouette.data;
	}

	// parallel extract of HOG features
	int freeCandidate = 0;
	#pragma omp parallel 
	{
		int id;
		while (true)
		{
			#pragma omp critical 
			{
				id = freeCandidate;
				freeCandidate++;
			}

			if (id >= numCandidates)
				break;
			//imwrite("a.jpg",candidates[id].silhouette);
			hog(silhouettePtrs[id], &featuresPtr[id*nPatchV*nPatchH*NBINS], nPatchV, nPatchH, srow, scol);
	
		}	
	}
	free(silhouettePtrs);
}


void CPointCloud::hog(double *silhouette, double *features, int nPatchV, int nPatchH, int srow, int scol) {

	// allocate memory for gradient and magnitude of gradient
	double *angles = (double *)calloc(srow*scol,sizeof(double));
	double *magnit = (double *)calloc(srow*scol,sizeof(double));
	// find maxDepth for silhouette normalization
	double maxDepth = 0, depth;
	for (int j=0; j<scol; j++) {
		for (int i=0; i<srow; i++) {
			depth = silhouette[i+j*srow];
			if ((int)depth != SDEPTH && depth > maxDepth) maxDepth = depth;
		}
	}

	double *silhouetteNormalized = (double *)malloc(sizeof(double)*srow*scol);
	memcpy(silhouetteNormalized, silhouette, sizeof(double)*srow*scol);

	// normalize silhouette
	for (int j=0; j<scol; j++) {
		for (int i=0; i<srow; i++) {
			if ((int)silhouette[i+j*srow] == SDEPTH )
				silhouetteNormalized[i+j*srow] = 0;
			else
				silhouetteNormalized[i+j*srow] = (maxDepth - silhouette[i+j*srow]);
		}
	}

	// compute gradient and gradient magnitude
	double dx, dy;
	for (int j=1; j<scol-1; j++) {
		for (int i=1; i<srow-1; i++) {
			dx = silhouetteNormalized[i + (j+1)*srow] - silhouetteNormalized[i + (j-1)*srow];
			dy = silhouetteNormalized[i-1 + j*srow] - silhouetteNormalized[i+1 + j*srow];
			angles[i+j*srow] = atan2(dy,dx);
			magnit[i+j*srow] = sqrt(dx*dx+dy*dy);
		}
	}

	// compute and normalize a histogram for every cell
	int histOffset;
	int patchNum = 0, numBin, id;
	double angle, sector = 2*CV_PI/NBINS, norm; 
	for (int j=0; j<nPatchH; j++) {
		for (int i=0; i<nPatchV; i++) {
			histOffset = patchNum*NBINS;
			for (int s=0; s<H_PATCH_SIZE; s++) {
				for (int t=0; t<V_PATCH_SIZE; t++) {
					id = i*V_PATCH_SIZE+t + (j*H_PATCH_SIZE+s)*srow;
					angle = angles[id]+CV_PI; // signed gradient, from 0 to 2PI
					numBin = (int)(angle/sector);
					if (numBin >= NBINS) numBin = NBINS-1;
					features[histOffset+numBin] += magnit[id];
				}
			}

			// block size is 1, normalize single cells
			norm = 0;
			for (int k=0; k<NBINS; k++) {
				norm += features[histOffset+k]*features[histOffset+k];
			}
			norm = sqrt(norm);

			for (int k=0; k<NBINS; k++) {
				if (norm > 1)
					features[histOffset+k] = features[histOffset+k]/norm;
				else
					features[histOffset+k] = 0; // histogram norm too small, clipping
			}

			patchNum++;
		}
	}
	free(silhouetteNormalized);
	free(angles);
	free(magnit);
}
void CPointCloud::dbscanClustering(Mat footCutPoints, int nNeigh, int radius, vector<int>&classes, vector<int>& types)
{
	int C = 0;
	int reserveSize = 50;
	int numPoints = footCutPoints.rows;
	double eps = radius;
	double k   = nNeigh;
	uint8_t *visited = (uint8_t *)calloc(numPoints,sizeof(uint8_t));

	double*footPtr = (double*)footCutPoints.data;
	PointCloud pc;
	pc.pts = footPtr;
	pc.numPoints = footCutPoints.rows;

	// create KDTree
	KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud>, PointCloud, DIM /* dim */> index(DIM /*dim*/, pc, KDTreeSingleIndexAdaptorParams(20 /* max leaf */) );
	index.buildIndex();
	const double search_radius = static_cast<double>(eps*eps);
	nanoflann::SearchParams params;
	params.sorted = false;
	double queryPoint[DIM];
	queue<int>neigh;
	vector<pair<size_t,double>> ret_matches;

	int n=0;
	int matches =0;
	//start clusterization
	for(int i=0; i<numPoints; i++)
	{
		if(visited[i] == 0)
		{
			queryPoint[0] = footPtr[3*i+0];//footCutPoints[i].x;
			queryPoint[1] = footPtr[3*i+1];//footCutPoints[i].y;
			queryPoint[2] = footPtr[3*i+2];//footCutPoints[i].z;

			ret_matches.clear();
			ret_matches.reserve(reserveSize);

			const size_t nMatches = index.radiusSearch(queryPoint, search_radius, ret_matches, params);
			if(nMatches == 1)
			{
				// outlier
				visited[i] = 1;
				classes[i] = -1;
				types[i]   = -1;
				n++;
			}
			else if(nMatches>k)
			{	 // core point - start expanding a new cluster
				n++;
				types[i] = 1;
				C = C + 1;	  /*class id of the new cluster*/
				visited[i] = 1;
				classes[i] = C;
				matches += nMatches;
				for(int j=0; j<nMatches; j++)
				{
					size_t idx = ret_matches[j].first;
					if(visited[idx] == 0 && idx!=i)
					{
						neigh.push(idx);  /* insert neighbors in the neighbors list */
						visited[idx] = 1;
						n++;
					}
				}
				// expand cluster until the neighbors list becomes empty
				while(!neigh.empty())
				{
					int id = neigh.front();
					neigh.pop();
					queryPoint[0] = footPtr[3*id+0];//footCutPoints[id].x;
					queryPoint[1] = footPtr[3*id+1];//footCutPoints[id].y;
					queryPoint[2] = footPtr[3*id+2];//footCutPoints[id].z;

					ret_matches.clear();
					ret_matches.reserve(reserveSize);

					// find the number of neighbors of current processed neighbor point 
					const size_t nMatches = index.radiusSearch(queryPoint, search_radius, ret_matches, params);
					matches += nMatches;
					if(nMatches < k+1)
					{  // border point
						types[id] = 0;
					}
					else
					{
						types[id] = 1;
						for(int j=0; j<nMatches; j++)
						{
							size_t idx =ret_matches[j].first;
							if(visited[idx] == 0 && idx!=id)
							{
								neigh.push(idx);
								visited[idx] = 1;
								n++;
							}
						}
					}
					if(classes[id] == 0)
						classes[id] = C;			
				}
			}

		}
	}
	free (visited);
	//delete points;
}

void CPointCloud::minBoundingBox(vector<Point3d>points, Point2f vertices[])
{
	vector<Point2f>pts;
	for(int i=0; i<points.size(); i++)
	{
		pts.push_back(Point2f(points[i].x, points[i].y));
	}

	RotatedRect box = minAreaRect(Mat(pts));
	box.points(vertices);
}


void CPointCloud::plotPC(Mat inv, Mat& imgOut, vector<PeopleTracks>peopleTracks)
{
	int sizeX = 640;
	int sizeY = 480;

	float fieldOfViewH = (58*1.0)/180*3.14;
	float fieldOfViewV = (45*1.0)/180*3.14;

	float fx = 320/tan(fieldOfViewH/2)*0.88;
	float fy = 240/tan(fieldOfViewV/2)*0.86;
	//camera extrinsic parameters
	Mat K = (Mat_<double>(3,3)<<fx, 0, sizeX/2,0, fy, sizeY/2, 0,0,1);
	Mat P = inv.rowRange(0,3).clone();
	Mat R = K*P;

	imgOut = Mat::zeros(Size(640,480),CV_8UC3);
	uchar* ptr= imgOut.data;
#pragma omp parallel for   
	for(int i=0; i<num_after_voxel; i++)
	{
		Point3d pt;
		pt.x =  R.at<double>(0,0)*rotatePointsPtr[3*i] + R.at<double>(0,1)*rotatePointsPtr[3*i+1] + R.at<double>(0,2)*rotatePointsPtr[3*i+2] + R.at<double>(0,3);
		pt.y =  R.at<double>(1,0)*rotatePointsPtr[3*i] + R.at<double>(1,1)*rotatePointsPtr[3*i+1] + R.at<double>(1,2)*rotatePointsPtr[3*i+2] + R.at<double>(1,3);
		pt.z =  R.at<double>(2,0)*rotatePointsPtr[3*i] + R.at<double>(2,1)*rotatePointsPtr[3*i+1] + R.at<double>(2,2)*rotatePointsPtr[3*i+2] + R.at<double>(2,3);

		int v = cvRound( sizeX - pt.x/pt.z );
		int u = cvRound( sizeY - pt.y/pt.z);

		if( v<=0||v>sizeX ||u<=0||u>sizeY)
			continue;

		imgOut.at<Vec3b>(u,v)[0] = colorsPtr[3*i];
		imgOut.at<Vec3b>(u,v)[1] = colorsPtr[3*i+1];
		imgOut.at<Vec3b>(u,v)[2] = colorsPtr[3*i+2];

	} 
	for(int i=0; i<peopleTracks.size(); i++)
	{
		
	//	double cx = peopleTracks[i].prediction.x.ptr<double>(0)[0]*1000;
	//	double cz = peopleTracks[i].prediction.x.ptr<double>(1)[0]*1000;
	//	for(int i=cx-200; i<cx+200; i++)
	//	{
	//		for(int j=cz-200; j<cz+200; j++)
	//		{
	//			Point3d pt;
	//			pt.x = R.at<double>(0,0)*i + R.at<double>(0,1)*j + R.at<double>(0,2)*7.5 + R.at<double>(0,3);
	//			pt.y = R.at<double>(1,0)*i + R.at<double>(1,1)*j + R.at<double>(1,2)*7.5 + R.at<double>(1,3);
	//			pt.z = R.at<double>(2,0)*i + R.at<double>(2,1)*j + R.at<double>(2,2)*7.5 + R.at<double>(2,3);

	//			double dcx = R.at<double>(0,0)*cx + R.at<double>(0,1)*cz + R.at<double>(0,2)*7.5 + R.at<double>(0,3);
	//			double dcy = R.at<double>(1,0)*cx + R.at<double>(1,1)*cz + R.at<double>(1,2)*7.5 + R.at<double>(1,3);
	//			double dcz = R.at<double>(2,0)*cx + R.at<double>(2,1)*cz + R.at<double>(2,2)*7.5 + R.at<double>(2,3);

	//			int mv = cvRound( sizeX - dcx/dcz  );
	//			int mu = cvRound( sizeY - dcy/dcz  );

	//			int v = cvRound( sizeX - pt.x/pt.z );
	//			int u = cvRound( sizeY - pt.y/pt.z);
	//		
	//			if( v<=0||v>sizeX ||u<=0||u>sizeY)
	//				continue;
	//			
	//			//double radius = sqrt((float)((mu-u) *(mu-u) + (mv-v)*(mv-v)));
	//		
	//		  
	//	
	//			imgOut.at<Vec3b>(u,v)[0] = peopleTracks[i].clr.val[0];
	//			imgOut.at<Vec3b>(u,v)[1] = peopleTracks[i].clr.val[1];
	//			imgOut.at<Vec3b>(u,v)[2] = peopleTracks[i].clr.val[2];	
	//			
	//		
	//			}	
	//			
	//	}

	
		vector<Point2f>pts;
		for(int j=0; j<peopleTracks[i].legs.size(); j++){

			double cx = peopleTracks[i].prediction.x.ptr<double>(0)[0]*1000;
			double cz = peopleTracks[i].prediction.x.ptr<double>(1)[0]*1000;
			
			Mat points = peopleTracks[i].legs[j].allPoints;		
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
		
					/*pts.push_back(Point2f(v, u));*/
						
					imgOut.at<Vec3b>(u,v)[0] = peopleTracks[i].clr.val[0];
					imgOut.at<Vec3b>(u,v)[1] = peopleTracks[i].clr.val[1];
					imgOut.at<Vec3b>(u,v)[2] = peopleTracks[i].clr.val[2];				
				}

			}
			
		}



		//RotatedRect box = minAreaRect(Mat(pts));
		///*Rect brect = box.boundingRect();
		//rectangle(imgOut, brect, Scalar(255,0,0));*/
		//Point2f vertices[4];
		//box.points(vertices);
		//for (int i = 0; i < 4; i++)
		//	line(imgOut, vertices[i], vertices[(i+1)%4], Scalar(0,0,255),2,8,0);
	}


}


float CPointCloud::deg2rad(float angleInDergees)
{
	return 	(CV_PI/180)*angleInDergees	;
}

int CPointCloud::nnz(double *pointPtr, PlaneParmeters plane, float tol)
{
	float thresh = sqrt((float)(plane.a*plane.a + plane.b*plane.b+ plane.c*plane.c));
	thresh = thresh * tol;
	int num = 0;
	for(int i=0; i<num_after_voxel; i++)
	{
		float val =abs( plane.a*pointPtr[3*i] +	 plane.b*pointPtr[3*i+1]  + plane.c*pointPtr[3*i+2] + plane.d*1);
		if(val<thresh)
		{
			num++;
		}
	}
	return num;
}

//eulr angle
void CPointCloud::rotationMatrixFromAxisAndAngle(int axisX, int axisY,int axisZ, float angle, Mat &ratationMatrix)
{

	int x = axisX;
	int y = axisY;
	int z = axisZ;
	float cosAngle = cos(angle);
	float sinAngle = sin(angle);
	ratationMatrix = (Mat_<double>(3,3)<<cosAngle+x*x*(1-cosAngle) ,  x*y*(1-cosAngle)-z*sinAngle ,  x*z*(1-cosAngle)+y*sinAngle,
										y*x*(1-cosAngle)+z*sinAngle,  cosAngle+y*y*(1-cosAngle),    y*z*(1-cosAngle)-x*sinAngle,
										z*x*(1-cosAngle)-y*sinAngle,  z*y*(1-cosAngle)+x*sinAngle,  cosAngle+z*z*(1-cosAngle));

}