import numpy as np
import matplotlib.pyplot as plt

def initObjPoints(a, ifPlot=False):
	n = 9
	objPoints = []

	marker0  = np.array([[-n+ 0, n- 0 ,0],
						 [-n+ 2, n- 0 ,0],
						 [-n+ 2, n- 2 ,0],
						 [-n+ 0, n- 2 ,0]])
	objPoints.append(marker0)

	marker1  = np.array([[-n+ 4, n- 0 ,0],
						 [-n+ 6, n- 0 ,0],
						 [-n+ 6, n- 2 ,0],
						 [-n+ 4, n- 2 ,0]])
	objPoints.append(marker1)

	marker2  = np.array([[-n+ 8, n- 0 ,0],
						 [-n+10, n- 0 ,0],
						 [-n+10, n- 2 ,0],
						 [-n+ 8, n- 2 ,0]])
	objPoints.append(marker2)

	marker3  = np.array([[-n+12, n- 0 ,0],
						 [-n+14, n- 0 ,0],
						 [-n+14, n- 2 ,0],
						 [-n+12, n- 2 ,0]])
	objPoints.append(marker3)

	marker4  = np.array([[-n+16, n- 0 ,0],
						 [-n+18, n- 0 ,0],
						 [-n+18, n- 2 ,0],
						 [-n+16, n- 2 ,0]])
	objPoints.append(marker4)

	marker5  = np.array([[-n+16, n- 4 ,0],
						 [-n+18, n- 4 ,0],
						 [-n+18, n- 6 ,0],
						 [-n+16, n- 6 ,0]])
	objPoints.append(marker5)

	marker6  = np.array([[-n+16, n- 8 ,0],
						 [-n+18, n- 8 ,0],
						 [-n+18, n-10 ,0],
						 [-n+16, n-10 ,0]])
	objPoints.append(marker6)

	marker7  = np.array([[-n+16, n-12 ,0],
						 [-n+18, n-12 ,0],
						 [-n+18, n-14 ,0],
						 [-n+16, n-14 ,0]])
	objPoints.append(marker7)

	marker8  = np.array([[-n+16, n-16 ,0],
						 [-n+18, n-16 ,0],
						 [-n+18, n-18 ,0],
						 [-n+16, n-18 ,0]])
	objPoints.append(marker8)

	marker9  = np.array([[-n+12, n-16 ,0],
						 [-n+14, n-16 ,0],
						 [-n+14, n-18 ,0],
						 [-n+12, n-18 ,0]])
	objPoints.append(marker9)

	marker10 = np.array([[-n+ 8, n-16 ,0],
						 [-n+10, n-16 ,0],
						 [-n+10, n-18 ,0],
						 [-n+ 8, n-18 ,0]])
	objPoints.append(marker10)

	marker11 = np.array([[-n+ 4, n-16 ,0],
						 [-n+ 6, n-16 ,0],
						 [-n+ 6, n-18 ,0],
						 [-n+ 4, n-18 ,0]])
	objPoints.append(marker11)

	marker12 = np.array([[-n+ 0, n-16 ,0],
						 [-n+ 2, n-16 ,0],
						 [-n+ 2, n-18 ,0],
						 [-n+ 0, n-18 ,0]])
	objPoints.append(marker12)

	marker13 = np.array([[-n+ 0, n-12 ,0],
						 [-n+ 2, n-12 ,0],
						 [-n+ 2, n-14 ,0],
						 [-n+ 0, n-14 ,0]])
	objPoints.append(marker13)

	marker14 = np.array([[-n+ 0, n- 8 ,0],
						 [-n+ 2, n- 8 ,0],
						 [-n+ 2, n-10 ,0],
						 [-n+ 0, n-10 ,0]])
	objPoints.append(marker14)

	marker15 = np.array([[-n+ 0, n- 4 ,0],
						 [-n+ 2, n- 4 ,0],
						 [-n+ 2, n- 6 ,0],
						 [-n+ 0, n- 6 ,0]])
	objPoints.append(marker15)

	objPoints = np.vstack(objPoints).astype('float32')
	objPoints = objPoints*a / (n*2)

	if ifPlot:
		for i in range(15):
			plt.plot(objPoints[i*4+0,0], objPoints[i*4+0,1], 'r.')
			plt.plot(objPoints[i*4+1,0], objPoints[i*4+1,1], 'g.')
			plt.plot(objPoints[i*4+2,0], objPoints[i*4+2,1], 'b.')
			plt.plot(objPoints[i*4+3,0], objPoints[i*4+3,1], 'y.')
		plt.axis('equal')
		plt.show()
		exit()

	return objPoints

