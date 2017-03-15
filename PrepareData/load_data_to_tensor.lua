gm = require 'graphicsmagick'
require 'lfs'

local main_path = '/home/adam/data/bowl/'
local data_path = main_path..'resized_proportion_64x64/'

-- Prepare train set
classes = {}
local i = 1
for file in lfs.dir(data_path..'train') do
   if file ~= '.' and file ~= '..' then
      classes[i] = file
      i = i + 1
   end
end

table.sort(classes)

meta_train = {}

local j = 1
for i, cls in pairs(classes) do
   for file in lfs.dir(data_path..'train/'..cls) do
      if file ~= '.' and file ~= '..' then
         local filepath = data_path..'train/'..cls..'/'..file
         meta_train[j] = {fpath=filepath, f=file, class=cls, class_number=i-1}
         j = j + 1
      end
   end
end

local output_path = main_path..'torch/'

torch.save(output_path..'train_list.dat', meta_train)

-- Save images into a tensor
train_tensor = torch.ByteTensor(#meta_train, 64*64+1)
print(train_tensor:size())

for i, imginfo in pairs(meta_train) do
   local img = gm.Image(imginfo.fpath)
   local t = img:toTensor('byte', 'I', 'DHW')
   train_tensor:sub(i, i, 1, 64*64):copy(t)
   train_tensor:sub(i, i, 64*64+1, 64*64+1):fill(imginfo.class_number)
end

torch.save(output_path..'train.th7', train_tensor)

--[[
-- Prepare test set
meta_test = {}

local j = 1
for file in lfs.dir(data_path..'test') do
   if file ~= '.' and file ~= '..' then
      local filepath = data_path..'test/'..file
      meta_test[j] = {fpath=filepath, f=file}
      j = j + 1
   end
end

torch.save(output_path..'test_list.dat', meta_test)

-- Save images into a tensor
test_tensor = torch.ByteTensor(#meta_test, 64*64)
print(test_tensor:size())

for i, imginfo in pairs(meta_test) do
   local img = gm.Image(imginfo.fpath)
   local t = img:toTensor('byte', 'I', 'DHW')
   test_tensor:select(1, i):copy(t)
end

torch.save(output_path..'test.th7', test_tensor)
]]
