-- local functions = require "util.functions"
local torch = require "torch"
local io = require "io"
local DataLoader = torch.class('DataLoader')

require "image"

function DataLoader:__init(opt)
    -- data directory path
    self.data_dir = opt.data_dir
    -- image meta data file path
    -- self.image_meta_data_file = opt.image_meta_data_file
    -- image objects file
    self.image_objects_file = opt.image_objects_file
    -- image attributes file
    -- self.image_attributes_file = opt.image_attributes_file
    -- total number of pixels in image
    self.img_size = opt.img_size
    -- number of images given in each gradient step
    self.batch_size = opt.batch_size
    -- current starting image number
    self.image_number = 1
    -- number of total training images
    self.number_of_training_images = opt.num_training
    
    -- initially load the meta data file and parse the JSON data
    json = require('cjson')
    
    local image_meta_data_file_ptr = io.open(self.data_dir .. 'image_data.json')
    local image_objects_file_ptr = io.open(self.data_dir .. 'objects.json')
    
    local json_line = ""
    for line in image_meta_data_file_ptr:lines() do
        json_line = json_line .. line
    end
    print('Parsing image meta data file...\n');
    self.image_meta_data_json = json.decode(json_line);
    
    print(self.image_meta_data_json[1].url)
    
    json_line = ""
    for line in image_objects_file_ptr:lines() do
        json_line = json_line .. line
    end
    print('Parsing objects json file...\n')
    self.image_objects_json = json.decode(json_line);
end


-- get the number of classes in data
function DataLoader:get_number_classes() 
    tab = torch.load("objects")
    return tab
    
end

-- Load the classes from Img Labels and then save to a file 
function DataLoader:get_img_labels()
    local img_length =#self.image_objects_json

    
    tab = {};
    print (tab)
    for obj =1, img_length do
        print("Img number " .. obj)
        local objects_length = #self.image_objects_json[obj].objects
        for j = 1, objects_length do
            local word = self.image_objects_json[obj].objects[j].names
            local names_len = #word

            if(names_len ~= 0) then
                --    table.insert(tab,1,self.image_objects_json[obj].objects[j].names[1])    
                if tab[self.image_objects_json[obj].objects[j].names[1]] == nil then
                     tab[self.image_objects_json[obj].objects[j].names[1]] = 1
                else
                     tab[self.image_objects_json[obj].objects[j].names[1]] = tab[self.image_objects_json[obj].objects[j].names[1]] + 1
                end
                
            end
        end
        
        
        if (self.image_objects_json[obj]~=nil or self.image_objects_json[obj].objects~=nil or self.image_objects_json[obj].objects[1]~=nil or self.image_objects_json[obj].objects[1].names~=nil or self.image_objects_json[obj].objects[1].names[1]~=nil) then
            table.insert(tab,1,self.image_objects_json[obj].objects[1].names[1])            
            
        else
            print ("Empty")    
        end
        --]]
        --print (tab)
    end
    torch.save("objects",tab);
   return tab
end



-- load batch of images
function DataLoader:next_batch()
    if self.image_number >= self.number_of_training_images then
        self.image_number = 1
    end
    local i = self.image_number
    imgs, labels = self:load_images_and_labels(i, i+self.batch_size-1)

    self.image_number = self.image_number + self.batch_size
    return imgs, labels
end

function DataLoader:load_images_and_labels(start_img,end_img)
    local imgs = torch.zeros(end_img-start_img+1, 1, self.img_size, self.img_size)
    local labels = torch.zeros(end_img-start_img+1, self.img_size, self.img_size)
    
    for image_no = start_img, end_img do
        -- getting the relative file path from the url
        local image_path = self.image_meta_data_json[image_no].url
        
        local urllen = string.len(image_path)
        local last_slash_loc =  string.find(string.reverse(image_path), "/")
        
        image_path = string.sub(image_path, 1, urllen-last_slash_loc)
        local second_last_slash_loc =  string.find(string.reverse(image_path), "/")
        second_last_slash_loc = urllen - last_slash_loc - second_last_slash_loc + 2
        
        image_path = self.image_meta_data_json[image_no].url
        image_path = string.sub(image_path, second_last_slash_loc)
        image_path = self.data_dir .. image_path
        
        print('Image number ' .. image_no)
        print('Resizing to '.. self.img_size)
        
        imgs[image_no-start_img+1] = image.scale(image.load(image_path), self.img_size, self.img_size)
        labels[image_no-start_img+1] = self.image_objects_json[image_no].objects[1].names[1]
    end
    return imgs, labels
end