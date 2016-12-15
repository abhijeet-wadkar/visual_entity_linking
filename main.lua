require 'torch'
require 'nn'
require 'optim'
require 'utils.DataLoader'
require 'models.vgg'
require 'models.small_model'
-- require 'dpnn'
-- require 'hdf5'

local opts = require('utils.parseopt')

local function main()
  -- parse command line input
  local opt = opts.parse(arg)
  
  -- load meta data in memory
  local data_loader = DataLoader(opt)
  collectgarbage()
  print('Data loaded in memory...')
 
  -- data_loader:get_img_labels()
  total_classes = data_loader:get_number_classes()
 
  -- creating vggnet model
  model = create_small_model(total_classes)
  print(model)
  
  -- define criteria
  criterion = nn.ClassNLLCriterion()
 
  print('Training the network')
  batch_size = opt.batch_size
  number_iterations = opt.iterations
  num_batches = math.floor(data_loader:get_number_of_data_points()/opt.batch_size)
  for i = 1, number_iterations*num_batches do
    imgs, labels  = data_loader:next_batch()
    collectgarbage()
    print(imgs:size())
    print(labels:size())
    model:zeroGradParameters()
    output = model:forward(imgs)
    loss = criterion:forward(output, labels)
    gradient = criterion:backward(output, labels)
    model:backward(imgs, gradient)
    print("Loss: " ..loss)
  end

end

main()
