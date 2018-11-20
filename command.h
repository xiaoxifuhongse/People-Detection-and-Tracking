#include "serialPort.h"
#include <iostream>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>  
#include <boost/function.hpp>
using namespace  std;


class CCommand
{
public: 
	CCommand();
	~CCommand();
public: 
	void moveContorl(int sumX, int sumY, int lastSeen);
	void rationControl(int sumX, int sumY, int lastSeen);
protected:
	void moveFoward();
	void moveBackwoard();
	void stop();
	void turnLeft();
	void turnRight();

	CSerial serialPort;
	



};