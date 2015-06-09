require 'torch'
require 'image'

local lfs = require 'lfs'
local __ = require 'moses'



local function get_filenames(folder)
   local file_names = {}
   for file in lfs.dir(folder) do
      if lfs.attributes(file,'mode') ~= 'directory' then
         table.insert(file_names, file)
      end
   end
   return file_names
end

local function get_images(folder)
   print('  Loading images from ' .. folder .. ' ...')
   local images = {}
   local im_names = get_filenames(folder)
   for _, im_name in ipairs(im_names) do
      local img = image.load(folder .. im_name, nil, 'byte')
      table.insert(images, img)
   end
   return images
end

local function merge_tables(t1, t2)
   for k,v in ipairs(t2) do
      table.insert(t1, v)
   end
   return t1
end

local function repeat_value(value, n)
   local tbl = {}
   for i = 1, n do
      table.insert(tbl, value)
   end
   return tbl
end


local function pack_data(tensor_tbl)
   -- pack images in single tensor
   local packed = torch.CharTensor()
   local ndims = tensor_tbl[1]:dim()
   local channels = ((ndims == 2) and 1) or tensor_tbl[1]:size(1)
   local height = tensor_tbl[1]:size(ndims - 1)
   local width = tensor_tbl[1]:size(ndims)
   packed:resize(#tensor_tbl, channels, height, width)
   for i,img in ipairs(tensor_tbl) do
      packed[i] = img
   end
   return packed
end


local function get_dataset(data_lists, classes)
   print('Packing data...')
   local feature_list = {}
   local label_list = {}
   for _, zipped in ipairs(__.zip(data_lists, classes)) do
      local data_list = zipped[1]
      local label = zipped[2]

      feature_list = merge_tables(feature_list, data_list)
      label_list = merge_tables(label_list, repeat_value(label, #data_list))
      -- label_list = merge_tables(label_list, torch.CharTensor(#data_list):fill(label))
   end
   local data = pack_data(feature_list)
   local labels = torch.ByteTensor(label_list)
   print('Done.')
   return { data=data, labels=labels }
end


-- function download_archive(remote_path)
--       local tar = paths.basename(remote_path)
--       os.execute('wget ' .. remote_path .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
-- end


local image_loader = {}
function image_loader.getDataset(folders, classes)
   print('Getting data...')
   local data_lists = {} 
   for i, folder in ipairs(folders) do
      table.insert(data_lists, get_images(folder))
   end
   print('Done.')
   return get_dataset(data_lists, classes)
end

local archive_loader = {}
function archive_loader.getDataset(path)
   local dataset = torch.load(path, 'ascii')
   return dataset
end



local function copy (t) -- shallow-copy a table
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do target[k] = v end
    setmetatable(target, meta)
    return target
end



local function upgrade_dataset(dataset)
   function dataset:size()
      return self.data:size(1)
   end

   function dataset:subset(from, n)
      local data = self.data:narrow(1, from, n)
      local labels = self.labels:narrow(1, from, n)

      local subset = copy(self)
      subset.data = data
      subset.labels = labels

      return subset
   end

   setmetatable(
      dataset, 
      { __index = function(self, index)
                     local input = self.data[index]
                     local label = self.labels[index]
                     return {input, label}
                  end
      }
   )
end



local loader = {}

function loader.loadDatasetFromArchive(path)
   dataset = archive_loader.getDataset(path)
   upgrade_dataset(dataset)
   return dataset
end

function loader.loadDataset(folders, classes, force, serialization_path)
   local dataset = nil
   local path = serialization_path or lfs.currentdir() .. '/' .. 'stored_dataset'
   if not force then
      status, msg = lfs.attributes(path, true)
      if status then
         print('Loading data from prepared storage (' .. path .. ') ...')
         dataset = torch.load(path)
         print('Done')
      end
   end
   if not dataset then
      dataset = image_loader.getDataset(folders, classes)
      print('Saving data...')
      torch.save(path, dataset)
      print('Done.')
   end

   upgrade_dataset(dataset)
   return dataset
end

return loader