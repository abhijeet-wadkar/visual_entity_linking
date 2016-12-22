require 'torch'
require 'nn'
cjson = require 'cjson'
require 'optim'
require 'utils.DataLoader'
opts = require "utils.parseopt"
require "math"

torch.manualSeed(10)

-- Paths and parameters
opt = opts.parse(arg)
opt.op_dir = 'Results/results10/'
chkpt = 'models/checkpoints'..opt.checkpointno..'/'..opt.test_model
print(chkpt)

model_learned = torch.load(chkpt)

-- Load Classes
loader = DataLoader(opt)
loader.number_of_training_images = math.floor(loader.total_images*.80)
loader.total_images = loader.total_images + 500
loader.batch_size = 1

count = 0

for i=1, 500 do

    imgs, labels  = loader:next_testing_batch()
    
    output = model_learned:forward(imgs[1])
 
    max_op, indices = torch.sort(output, true)

    print(#labels)
    print(#indices)

    local correct = 0
    local no_correct = 0
    for j = 1, 5 do
      if labels[1][indices[j]] == 1 then
        correct = correct + 1
      else 
        no_correct = no_correct + 1
      end
    end

    if correct == 5 then
      count = count + 1
    end
    
    print('For image '.. i .. ' correct: ' .. correct .. 'non_correct: '..no_correct)
end

print('Correct predicted: ' .. count)
