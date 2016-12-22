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

  -- check for CUDA
  local ok,cunn = pcall(require,'cunn')
  local ok2,cutorch = pcall(require,'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 and opt.gpuid ~= -1 then
    print('using CUDA on GPU ')
    cutorch.setDevice(opt.gpuid)
  else
    print('Falling back on CPU mode')
  end  

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
  -- criterion = nn.ClassNLLCriterion()
  criterion = nn.MultiLabelSoftMarginCriterion()

  -- moving model and criteria on GPU
  if ok and ok2 and opt.gpuid ~= -1 then
    model = model:cuda()
    criterion = criterion:cuda()
  end 

  print('Training the network')
  batch_size = opt.batch_size
  number_iterations = opt.iterations
  num_batches = math.floor(data_loader:get_number_of_data_points()/opt.batch_size)
  for i = 1, number_iterations*num_batches do
    -- get next batch of data
    imgs, labels  = data_loader:next_batch()

    collectgarbage()
    
    if ok and ok2 and opt.gpuid ~= -1 then
        imgs = imgs:cuda()
        labels = labels:cuda()
    end

    -- pass it through network and change weights
    model:zeroGradParameters()
    output = model:forward(imgs)
    loss = criterion:forward(output, labels)
    gradient = criterion:backward(output, labels)
    model:backward(imgs, gradient)
    print("Loss: " ..loss)
  end
end

main()
