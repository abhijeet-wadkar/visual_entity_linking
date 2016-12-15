
require 'torch'
require 'nn'
-- require 'dpnn'
-- require 'hdf5'
require 'optim'
require 'utils.DataLoader'
local opts = require('utils.parseopt')

local function main()
  
  -- parse command line input
  local opt = opts.parse(arg)
  
  -- load meta data in memory
  local data_loader = DataLoader(opt)
  collectgarbage()
  print('Data loaded in memory...')
  
  --imgs, labels  = data_loader:next_batch()
  t=data_loader:get_number_classes()
  print (t)
  print (#t) 
  
    
end

main()
