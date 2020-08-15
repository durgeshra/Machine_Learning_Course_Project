import numpy as np
import random as rnd
import time as tm
import math
# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def solver( X, y, C, timeout, spacing ):
	(n, d) = X.shape
	t = 0
	totTime = 0
	# w is the normal vector and b is the bias
	# These are the variables that will get returned once timeout happens
	w = np.zeros( (d,) )
	b = 0
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
	theta = np.append( w, b )
	normSq = np.square( np.linalg.norm( X, axis = 1 ) ) + 1
	alpha = np.zeros((n,))
	global randpermInner,randperm
	randperm = np.random.permutation( y.size )
	randpermInner = -1
	# You may reinitialize w, b to your liking here
	# You may also define new variables here e.g. eta, B etc

################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				return (w, b, totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

		if (C>0.1):
			(theta,alpha) = doSDCA(X,y,C,getRandpermCoord,getStepLength,alpha,t,w,b,min(100,X.shape[0]))
		else:
			(theta,alpha) = doSDCM_average(X,y,C,getRandCoord,alpha,normSq,t, theta,min(200,X.shape[0]))
		w = theta[:-1]
		b = theta[-1]
		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses - severe penalties await
		
		# Please note that once timeout is reached, the code will simply return w, b
		# Thus, if you wish to return the average model (as we did for GD), you need to
		# make sure that w, b store the averages at all times
		# One way to do so is to define two new "running" variables w_run and b_run
		# Make all GD updates to w_run and b_run e.g. w_run = w_run - step * delw
		# Then use a running average formula to update w and b
		# w = (w * (t-1) + w_run)/t
		# b = (b * (t-1) + b_run)/t
		# This way, w and b will always store the average and can be returned at any time
		# w, b play the role of the "cumulative" variable in the lecture notebook
		# w_run, b_run play the role of the "theta" variable in the lecture notebook
		
	return (w, b, totTime) # This return statement will never be reached

def getStepLength( grad, t ):
	return 0.5/(math.sqrt(t+1))

def getRandCoord( currentCoord,n):
	return rnd.randint( 0, n-1 )

def getRandpermCoord( currentCoord,n):
	global randperm, randpermInner
	if randpermInner >= n-1 or randpermInner < 0 or currentCoord < 0:
		randpermInner = 0
		randperm = np.random.permutation( n )
		return randperm[randpermInner]
	else:
		randpermInner = randpermInner + 1
		return randperm[randpermInner]

def doSDCA(X,y,C,getCoordFunc,stepFunc, init,t,w_avg,b_avg,B):
	totTime = 0
	
	# Initialize model as well as some bookkeeping variables
	alpha = init
	alphay = np.multiply( alpha, y )
	# Initialize the model vector using the equations relating primal and dual variables
	w = X.T.dot( alphay )
	# Recall that we are imagining here that the data points have one extra dimension of ones
	# This extra dimension plays the role of the bias in this case
	b = alpha.dot( y )
	tic = tm.perf_counter()
	samples = rnd.sample( range(0, X.shape[0]), B )

	for i in samples:
		x = X[i,:]
		#Find the gradient w.r.t Alpha
		delta = 1 - alpha[i]/(2*C)- y[i]*(x.dot(w) + b)

		newAlphai =  alpha[i] + stepFunc( delta, t+1 ) * delta
		#         print(newAlphai)
		# Make sure that the constraints are satisfied
		if newAlphai < 0:
			newAlphai = 0

		# Update the model vector and bias values
		# Takes only O(d) time to do so
		w = w + (((newAlphai - alpha[i])) * y[i] * x)
		b = b + (((newAlphai - alpha[i])) * y[i])
		alpha[i] = newAlphai
	w_avg = (w_avg*(t-1) + w)/(t)
	b_avg = (b_avg*(t-1) + b)/(t)
	toc = tm.perf_counter()
	totTime = totTime + (toc - tic)

#     print( "nSV = ", np.sum( alpha > C/100 ), " out of ", y.size, "data points" )    
	return (np.append( w_avg, b_avg ),alpha)#, primalObjValSeries, dualObjValSeries, timeSeries)

def doSDCM_average(X,y,C,getCoordFunc, init,normSq, iteration_no, theta,B):
	totTime = 0
	n,d = X.shape
	# Initialize model as well as some bookkeeping variables
	alpha = init
	alphay = np.multiply( alpha, y )
	# Initialize the model vector using the equations relating primal and dual variables
	w_run = X.T.dot( alphay )
	w_avg = theta[:-1]
	# w = w_run
	# Recall that we are imagining here that the data points have one extra dimension of ones
	# This extra dimension plays the role of the bias in this case
	b_run = alpha.dot( y )
	b_avg = theta[-1]
	# b = b_run
	# Calculate squared norms taking care that we are appending an extra dimension of ones
	
	# We have not made any choice of coordinate yet
	
	samples = rnd.sample( range(0, n), B )
	
	tic = tm.perf_counter()
	for i in samples:
		
		i = getCoordFunc( i,n )
		x = X[i,:]
		
		# Find the unconstrained new optimal value of alpha_i
		newAlphai =  (1 - y[i] * (x.dot(w_run) + b_run - alphay[i]*normSq[i])) / (normSq[i] + 1/(2*C))
		# Make sure that the constraints are satisfied
		if newAlphai < 0:
			newAlphai = 0
		
		# Update the model vector and bias values
		# Takes only O(d) time to do so
		w_run = w_run + ((newAlphai - alpha[i]) * y[i] * x)
		b_run = b_run + ((newAlphai - alpha[i]) * y[i])
		alpha[i] = newAlphai
	w_avg = (w_avg*(iteration_no) + w_run)/(iteration_no+1)
	b_avg = (b_avg*(iteration_no) + b_run)/(iteration_no+1)
	toc = tm.perf_counter()
	totTime = totTime + (toc - tic)
	return (np.append( w_avg, b_avg ),alpha)