require 'nn'
 
function create_small_model(number_of_categories)

  model = nn.Sequential()
  
  model:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
  model:add(nn.ReLU())                       -- non-linearity 
  model:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
  
  model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())                       -- non-linearity 
  model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())                       -- non-linearity 
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())                       -- non-linearity
  model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())                       -- non-linearity
  model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())  
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  model:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())  
  model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())  
  model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())  
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())  
  model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())  
  model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())  
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  model:add(nn.View(512*7*7))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
  model:add(nn.Linear(512*7*7, 120))             -- fully connected layer (matrix multiplication between input and weights)
  model:add(nn.ReLU())                       -- non-linearity 
  model:add(nn.Linear(120, 84))
  model:add(nn.ReLU())                       -- non-linearity 
  model:add(nn.Linear(84, #number_of_categories))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
  model:add(nn.LogSoftMax())
  
  return model
end         
