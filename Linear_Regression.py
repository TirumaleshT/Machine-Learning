import numpy 

def compute_error_for_line_given_points(b, m, points):
    #inital value of error
    total_error = 0
    #get every x and v point
    
    for i in range(len(points)):
        #get x value from data
        (x, y) = points[i]
		
        #get y value from data
        #y = points[i, 1]
        #calculate square of error i.e, (actual - predicted value)^2
        total_error += (y - (m * x + b)) **2
    
    #retugn the average of the total squared error
    return total_error / float(len(points))

#-------------------------------------------------------------------------
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    #starting b, m
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        #update b and m for every iteration with more accurate b and m by performing gradient step
        b, m = step_gradient(b, m, numpy.array(points), learning_rate)
    return [b, m]
#------------------------ -------------------------------------------------
def step_gradient(b_current, m_current, points, learningRate):
    
    #starting values for m, b gradients
    m_gradient = 0
    b_gradient = 0
    N = float(len(points))
    
    for i in range(len(points)):
        (x, y) = points[i]
        
        #now we will calculate the m and b gradient values to minimise the error
        m_gradient += -(2/N)* x * (y - ((m_current* x) + b_current))
        b_gradient += -(2/N)* (y - (m_current * x) + b_current)
        
    #update b and m values
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    
    return [new_b, new_m]
	
#--------------------------------------------------------------------------
def run():
    #Step 1 :load dataset 
    points = numpy.genfromtxt('train.csv', delimiter = ',', names  = True)
    #print(points.ndim)
	#define hyper parameters
    #learning rate
    learning_rate = 0.01
    #linear regression line y = mx + b
    initial_m = 0      #initial slope
    initial_b = 0      #initial y intercept
    num_iterations = 100
    
    #Step 3 Train our model
    print('starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print( 'ending gradient descent at b = {1}, m = {2} and error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == "__main__":
    run()
