from math import *
import random
import numpy as np
from treXtonConfig import parser
import cv2
import matplotlib.pyplot as plt

args = parser.parse_args()
landmarks  = [[20.0, 20.0], 
              [80.0, 80.0],
              [20.0, 80.0],
              [80.0, 20.0]]
world_img = cv2.imread(args.mymap)
world_h, world_w, world_c = world_img.shape

xs = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 2000]
ys = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 0, 1800, 2000, 2200]


def calc_dist(x, y):
    x_np = np.array(x)
    y_np = np.array(y)
    return np.linalg.norm(x_np - y_np)


def angle_between(p1_in, p2_in):
    
    diff = p2_in - p1_in
    ang = np.arctan2(diff[1], diff[0])
    return ang % (2 * np.pi)

class robot:
    def __init__(self):
        self.x = random.random() * world_w
        self.y = random.random() * world_h
        self.orientation = random.random() * 2.0 * pi
        self.speed = random.random()
        self.forward_noise = 0.0;
        self.turn_noise    = 0.0;
        self.sense_noise   = 0.0;
        self.speed_noise   = 0.0;
        self.t = 0
    
    def set(self, new_x, new_y, new_orientation, new_speed):
        check_errors = False
        if check_errors:
            if new_x < 0 or new_x >= world_size:
                raise ValueError, 'X coordinate out of bound'
            if new_y < 0 or new_y >= world_size:
                raise ValueError, 'Y coordinate out of bound'
            if new_orientation < 0 or new_orientation >= 2 * pi:
                raise ValueError, 'Orientation must be in [0..2pi]'
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
        self.speed = float(new_speed)
    
    
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise, new_v_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.forward_noise = float(new_f_noise);
        self.turn_noise    = float(new_t_noise);
        self.sense_noise   = float(new_s_noise);
        self.speed_noise = float(new_v_noise);
    
    
    def sense(self, xy):
        Z = np.array([xy[0], xy[1]])

        print(Z)

        # TODO: specify dt
        t = self.t # save
        particle = self.move(Z[0], Z[1], 1)
        particle.t = t + 1

        return particle
    
    
    def move(self, new_x, new_y, dt):
 
        delta_x = new_x - self.x
        delta_y = new_y - self.y

        #turn = atan2(delta_y, delta_x) + pi
        forward = calc_dist(np.array([self.x, self.y]), 
                                     np.array([new_x, new_y]))

        # TODO: Think about a turn model
        
        # turn, and add randomness to the turning command
        # Thrun:
        # orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        # Me:
        turn = angle_between(np.array([new_x, new_y]),
                             np.array([self.x, self.y])) - (pi / 1.0)

        orientation = self.orientation + (float(turn) - self.orientation) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * pi
        
        # move, and add randomness to the motion command
        measured_dist = float(forward)
        pred_dist = float(self.speed * dt)

        measured_weight = 0.2
        pred_dist_weight = 1 - measured_weight

        dist = measured_weight * measured_dist + pred_dist * pred_dist_weight + random.gauss(0.0, self.forward_noise)
        #print("turn", turn)
        #print("prev x", self.x)
        #print("new x", new_x)
        #print("cos", cos(orientation))

        #print("prev y", self.y)
        #print("new y", new_y)
        #print("sin", sin(orientation))

        #print("dist is", dist)

        x = self.x + (cos(orientation) * dist)
        y = self.y + (sin(orientation) * dist)
        #print("x", x)
        #print("y", y)
        #print("")

        speed = dist * dt # + max(0, random.gauss(0.0, self.speed_noise))
        #print("speed is", dist * dt)

        # set particle
        res = robot()
        res.set(x, y, orientation, speed)
        res.set_noise(self.forward_noise, 
                      self.turn_noise, 
                      self.sense_noise, 
                      self.speed_noise)
        return res
    
    def Gaussian(self, mu, sigma, x):
        
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
    
    
    def measurement_prob(self, measurement, dt):
        
        # calculates how likely a measurement should be
        
        prob = 1.0;
        pred_forward = self.speed * dt

        pred_x = self.x + (cos(self.orientation) * pred_forward)
        pred_y = self.y + (sin(self.orientation) * pred_forward)
        pred = np.array([pred_x, pred_y])

        #print("measurement", measurement)
        
        dist = calc_dist(pred, measurement)

        prob *= self.Gaussian(pred_forward, self.sense_noise, dist)
        return prob
        
    
    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s speed=%.6s]' % (str(self.x), str(self.y), str(self.orientation), str(self.speed))


def eval(r, p):
    sum = 0.0;
    for i in range(len(p)): # calculate mean error
        dx = (p[i].x - r.x + (world_size/2.0)) % world_size - (world_size/2.0)
        dy = (p[i].y - r.y + (world_size/2.0)) % world_size - (world_size/2.0)
        err = sqrt(dx * dx + dy * dy)
        sum += err
    return sum / float(len(p))

def init_particles(num_particles):
    p = []

    f_noise = 200
    t_noise = 0.5
    s_noise = 200
    v_noise = 10

    for i in range(num_particles):
        r = robot()
        r.set_noise(f_noise, 
                    t_noise, 
                    s_noise, 
                    v_noise)
        p.append(r)

    return p
        

# Make a robot called myrobot that starts at
# coordinates 30, 50 heading north (pi/2).
# Have your robot turn clockwise by pi/2, move
# 15 m, and sense. Then have it turn clockwise
# by pi/2 again, move 10 m, and sense again.
#
# Your program should print out the result of
# your two sense measurements.
#
# Don't modify the code below. Please enter
# your code at the bottom.

####   DON'T MODIFY ANYTHING ABOVE HERE! ENTER CODE BELOW ####

#myrobot.set(30, 50, pi / 2.)
#myrobot = myrobot.move(pi / 2., 15)
#print myrobot.sense()
#myrobot = myrobot.move(pi / 2., 10)
#print myrobot.sense()

def move_all(p, xy, dt):

    """
    Move all particles
    """

    ps = []
    for i in range(len(p)):
        particle = p[i].sense(xy)
        ps.append(particle)

    return ps
        

def get_weights(p, xy, dt, t):
    
    """
    Get fitness for a list of particles
    """

    ws = []
    w_sum = 0
    for i in range(len(p)):
        w = p[i].measurement_prob(xy, dt)
        #w = p[i].measurement_prob(np.array([xs[t], ys[t]]), dt)
        w_sum += w
        ws.append(w)

    return ws, w_sum

def get_x_y(p):
    xs = []
    ys = []
    for i in range(len(p)):
        xs.append(p[i].x)
        ys.append(p[i].y)
    
    return np.array([xs, ys])
        


def resample_wheel(p, w, N):

    """
    Use Sebastian Thrun's resampling wheel
    p: particles
    w: weights
    N: Desired number of particles
    """

    new_p = []
    idx = int(random.random() * N)
    beta = 0.0
    mw = max(w)

    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > w[idx]:
            beta -= w[idx]
            idx = (idx + 1) % N
        new_p.append(p[idx])
    
    return new_p
    

if __name__ == "__main__":

    myrobot = robot()
    N = 100  # Number of particles
    T = 13  # Time steps
    p = init_particles(N)
    dt = 1

    for t in range(T):
        p = move_all(p, dt)
        plt_xs, plt_ys = get_x_y(p)
        fig, ax = plt.subplots()
        ax = fig.add_subplot(111)
        ax.scatter(plt_xs, plt_ys)
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, 4000)
        plt.show()
        ws, w_sum = get_weights(p, dt, t)
        new_p = resample_wheel(p, ws, N)
        for i in new_p: print t, i


    

                
    
