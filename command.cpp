#include "command.h"
static int mf = 1;
static int mb = 1;
static int tr = 1;
static int tl = 1;
static int  moveType = 1;
boost::mutex mu;
CCommand::CCommand()
{
	serialPort.Open(5,115200);
}
CCommand::~CCommand()
{
	
}
void CCommand::moveContorl(int sumX, int sumY, int lastSeen)
{
	if(sumX == 0 || sumY == 0)
	{
		stop();
	}

	mu.lock();
	cout<<sumY<<endl;
	if(moveType == 2){

		if(sumY>420 && lastSeen==0) //ºóÍË
		{
			moveFoward();
			
		}
		else if(sumY<400 && lastSeen==0)//Ç°½ø
		{
			
			moveBackwoard();
		}
		else
		{
			stop();
		}
	}
	mu.unlock();


}
void CCommand::rationControl(int sumX, int sumY, int lastSeen)
{

	mu.lock();
	if(moveType == 1){
		double angle = atan((double)abs(sumX - 320)/(640-sumY))*180/3.14;
		if(angle>5 && (sumX-320)>0 && lastSeen==0)
		{
			turnLeft();

		}
		else if(angle>5 && (sumX-320)<0 && lastSeen==0)
		{
			turnRight();

		}
		else
		{
			stop();
		}
	}
	mu.unlock();
}

void CCommand::moveFoward()
{
	if(mf == 1)
	{
		cout<<"forward"<<endl;
		//char cmd[9] = {0xaa, 0x55, 0x01, 0x00, 0x00, 0x02, 0x04, 0x6c, 0x20};
		//serialPort.SendData(cmd, 9);
		char cmd[11] = {0xAA, 0x55, 0x01 ,0x00, 0x02, 0x02, 0x05, 0x0B, 0xFF, 0x9D, 0x76};
		serialPort.SendData(cmd, 11);
		Sleep(50);
		mb = 1;
	}
	mf++;
}


void CCommand::moveBackwoard()
{
	if(mb == 1)
	{	cout<<"backward"<<endl;
		mf = 1;
		//char cmd[9] = {0xaa, 0x55, 0x01, 0x00, 0x00, 0x02, 0x06, 0xad, 0xa1};
		//serialPort.SendData(cmd, 9);
		char cmd[11] = {0xAA, 0x55, 0x01, 0x00, 0x02, 0x02, 0x07, 0x10, 0xFF, 0xAD,0xDD};
		serialPort.SendData(cmd, 11);
		Sleep(50);
	}
	mb++;
}

void CCommand::turnLeft()
{
	if(tl == 1)
	{

		tr = 1;
		//char cmd[9] = {0xaa, 0x55, 0x01, 0x00, 0x00, 0x02, 0x00, 0xaf, 0x21};
		char cmd[11] = {0xAA ,0x55, 0x01, 0x00, 0x02, 0x02, 0x01, 0x01, 0x30, 0xa8, 0x71};
		serialPort.SendData(cmd, 11);
		
	
	}
	tl++;
}

//AA 55 01 00 02 02 02 00 50 10 80
void CCommand::turnRight()
{
	if(tr == 1)
	{
		cout<<"right"<<endl;
		tl = 1;
		char cmd[11] = {0xAA, 0x55, 0x01, 0x00, 0x02, 0x02, 0x02, 0x00, 0x10, 0xe0,0x81};
		//char cmd[9] = {0xaa, 0x55, 0x01, 0x00, 0x00, 0x02, 0x02, 0x6e, 0xa0};
		serialPort.SendData(cmd,11);
	
	}
	tr++;

}

void CCommand::stop()
{
	tr = 1;
	tl = 1;
	mf = 1;
	mb = 1;
	
	char cmd[9] = {0xAA, 0x55, 0x01, 0x00, 0x00, 0x02, 0x08, 0x69, 0x20};
	serialPort.SendData(cmd, 9);
	if(moveType == 1) moveType = 2;
	else
		moveType = 1;
}

//void CCommand::crc16(unsigned char *ptr, unsigned int len)
//{
//	unsigned long wcrc = 0XFFFF;
//	unsigned char temp;
//	
//	for(int i=0; i<len; i++)
//	{
//		temp = (*ptr) & 0X00FF;
//		ptr++;
//		wcrc ^= temp;
//		for(int j=0; j<8; j++)
//		{
//			if(wcrc & 0X0001)
//			{
//				wcrc >>= 1;
//				wcrc ^= 0XA001;
//			}
//			else
//			{
//				wcrc >>= 1;
//			}
//		}
//	}
//	//temp=wcrc;
//	CRC[0] = wcrc;
//	CRC[1] = wcrc >> 8;
//}