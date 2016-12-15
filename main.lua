
require 'torch'
require 'nn'
require 'optim'
require 'utils.DataLoader'
require 'models.vgg'
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
  
  -- creating vggnet model
  model = create_vgg_model(1000)
  print(model)
  
  -- define criteria
  criterion = nn.ClassNLLCriterion()
  
  batch_size = opt.batch_size
  number_iterations = opt.iterations
  num_batches = math.floor(opt.num_training/opt.batch_size)
  for i = 1, number_iterations*num_batches do
    imgs, labels  = data_loader:next_batch()
    model:zeroGradParameters()
    output = model:forward(imgs)
    loss = criterion:forward(output, labels)
    gradient = criterion:backward(output, labels)
    model:backward(imgs, gradient)
    print("Loss: " ..loss)
  end
    
end

main()
