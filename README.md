# Theano_Tile_Coding
A tile coder in theano for Reinforcement Learning tasks

# Experiments
## 2D surface approximation
As a simple toy task, we try to approximate the function z = sin(x) + cos(y). over the range of [0, 2 pi] for both x and y dimensions.

![](https://github.com/mohammadpz/Theano_Tile_Coding/blob/master/files/2d_example.gif)

## Motion modeling
![](https://github.com/mohammadpz/Theano_Tile_Coding/blob/master/files/mocap.png)

We test our framework on an adopted version of the Horde model. Horde is a multi-timescale future prediction model on an RL robot that uses thousands of parallel learners in order to approximate many different outputs at the same time. The task of continually prediction future in different time-scales is called "Nexting". We adopted this model for modeling human motion. We used a single walking sequence from MIT MoCap dataset and trained our adopted version of Horde on it, armed with our Theano-based implementation of Tile Coding. The data consists of ~3000 time-steps and in each time-step position of 17 body joint is presented. All body joints are represented in 49-dimensional space where each dimension is normalized to be in $[0, 1]$. The task is to predict future of each joint in the next 10 time-steps given the current position of all joints. The feature space is tile-coded into 30 tilings with 50 tiles each. An example of a trained model predictions for one of the joints is shown in Figure below,

![](https://github.com/mohammadpz/Theano_Tile_Coding/blob/master/files/description.png)

![](https://github.com/mohammadpz/Theano_Tile_Coding/blob/master/files/final.gif)
