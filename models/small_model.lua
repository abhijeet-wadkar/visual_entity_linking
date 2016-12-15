require 'nn'
 
function create_small_model(number_of_categories)

  model = nn.Sequential()
  model:add(nn.SpatialConvolution(3, 6, 5, 5, 1, 1, 2, 2)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
  model:add(nn.ReLU())                       -- non-linearity 
  model:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
  model:add(nn.SpatialConvolution(6, 16, 5, 5, 1, 1, 2, 2))
  model:add(nn.ReLU())                       -- non-linearity 
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  model:add(nn.SpatialConvolution(16, 32, 5, 5, 1, 1, 2, 2))
  model:add(nn.ReLU())                       -- non-linearity 
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  model:add(nn.View(32*4*4))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
  model:add(nn.Linear(32*4*4, 120))             -- fully connected layer (matrix multiplication between input and weights)
  model:add(nn.ReLU())                       -- non-linearity 
  model:add(nn.Linear(120, 84))
  model:add(nn.ReLU())                       -- non-linearity 
  model:add(nn.Linear(84, #number_of_categories))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
  model:add(nn.LogSoftMax())
  
  return model
end         