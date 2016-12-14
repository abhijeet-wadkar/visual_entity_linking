local torch = require "torch"
local M = { }

function M.parse(arg)

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Train on Genome dataset')
    cmd:text()
    cmd:text('Options')

    cmd:option('-data_dir', './', 'path to directory with data')
    cmd:option('-img_size', 1280,'Image Size')
    cmd:option('-num_training', 80000, 'Number of training Examples')
    cmd:option('-batch_size', 20, 'Batch Size')
    cmd:option('-iterations', 100, 'Number of epochs')
    cmd:option('-learningRate', 0.01, 'learningRate')
    cmd:option('-learning_rate_decay', 0.95, 'learning_rate_decay')
    cmd:option('-learning_rate_decay_after', 20, 'learning_rate_decay')
    cmd:option('-gpuid',2, 'which gpu to use. -1 = use CPU')
    cmd:option('-num_val',2000, 'Number of validation images')
    cmd:option('-checkpoint_every',2, 'Checkpoint after iterations')
    cmd:option('-checkpointno',10, 'Directory for storing checkpoints')
    cmd:option('-test_model',32000, 'Number of validation images')

    cmd:text()

    local opt = cmd:parse(arg or {})

    opt.img_file_path = opt.data_dir.."images/"
    opt.masks_file_path = opt.data_dir .. "labels/"
    opt.json_file_path = opt.data_dir .. "Details.txt"

    print(opt)
    return opt
end

return M