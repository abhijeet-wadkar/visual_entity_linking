local functions = require "util.functions"
local torch = require "torch"
local io = require "io"
local DataLoader = torch.class('DataLoader')

require "image"

function DataLoader:__init(opt)
    -- data directory path
    self.data_dir = opt.data_dir
    -- image meta data file path
    self.image_meta_data_file = opt.image_meta_data_file
    -- image objects file
    self.image_objects_file = opt.image_objects_file
    -- image attributes file
    self.image_attributes_file = opt.image_attributes_file
    -- total number of pixels in image
    self.img_size = opt.img_size
    -- number of images given in each gradient step
    self.batch_size = opt.batch_size
    -- current starting image number
    self.image_number = 0
    -- number of total training images
    self.number_of_training_images = opt.number_of_training_images
    
    -- initially load the meta data file and parse the JSON data
    json = require('json')
    
    local image_meta_data_file_ptr = io.open(self.image_meta_data_file)
    local image_objects_file_ptr = io.open(self.image_objects_file)
    
    local json_line = ""
    for line in self.image_meta_data_file_ptr:lines() do
        json_line = json_line .. line
    end
    
    self.image_objects_json = json.decode(json_line);
     
end

-- get the number of classes in data
function DataLoader:get_number_classes() 
    return classes
end

-- load batch of images
function DataLoader:next_batch()
    if self.image_number >= self.number_of_training_images then
        self.image_number = 0
    end
    local i = self.image_number
    imgs, labels = self:load_images_and_labels(i, i+self.batch_size-1)

    self.image_number = self.image_number + self.batch_size
    return imgs, labels
end

function DataLoader:load_img_labels(start_img,end_img)
    local imgs = torch.zeros(end_img-start_img+1, 1, self.img_size, self.img_size)
    local labels = torch.zeros(end_img-start_img+1, self.img_size, self.img_size)
    for img_no = start_img, end_img do
        local image_path = self.data_dir .. "/" .. self.image_meta_data_json[image_no].url
        imgs[img_no-start_img+1] = image.load(image_path)
        labels[img_no-start_img+1] = self.image_objects_json[image_no]['objects']
    end
    return imgs, labels
end