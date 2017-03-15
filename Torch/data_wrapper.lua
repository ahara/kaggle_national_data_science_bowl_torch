-------------------------------------------------
--[[ NDSBowl ]]--
-------------------------------------------------
local Bowl, parent = torch.class("dp.Bowl", "dp.DataSource")
Bowl.isBowl = true

Bowl._name = 'Bowl'
Bowl._image_size = {64, 64, 1}
Bowl._feature_size = 1*64*64
Bowl._image_axes = 'bhwc'

function Bowl:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local load_all, input_preprocess, target_preprocess

   self.args, self._valid_ratio, self._shuffle, self._train_file, 
   self._test_file, self._data_path, self._scale, 
   load_all, input_preprocess, target_preprocess
   = xlua.unpack( 
   {config},
   'Bowl', nil,
   {arg='valid_ratio', type='number', default=1/5,
    help='proportion of training set to use for cross-validation.'},
   {arg='shuffle', type='boolean', default=true,
    help='shuffle train set before splitting into train/valid'},
   {arg='train_file', type='string', 
    default='train.th7', help='name of train_file'},
   {arg='test_file', type='string', 
    default='test.th7', help='name of test_file'},
   {arg='data_path', type='string', default='/home/adam/data/bowl/torch/',
    help='path to data repository'},
   {arg='scale', type='table', 
    help='bounds to scale the values between'},
   {arg='load_all', type='boolean', 
    help='Load all datasets : train, valid, test.', default=true},
   {arg='input_preprocess', type='table | dp.Preprocess',
    help='to be performed on set inputs, measuring statistics ' ..
    '(fitting) on the train_set only, and reusing these to ' ..
    'preprocess the valid_set and test_set.'},
   {arg='target_preprocess', type='table | dp.Preprocess',
    help='to be performed on set targets, measuring statistics ' ..
    '(fitting) on the train_set only, and reusing these to ' ..
    'preprocess the valid_set and test_set.'} 
   )
   if (self._scale == nil) then
      self._scale = {0,1}
   end
   self._classes = self:loadClasses()
   
   if load_all then
      self:loadTrainValid()
      --self:loadTest()
   end
   
   parent.__init(
      self, {
         train_set=self:trainSet(), valid_set=self:validSet(),
         --test_set=self:testSet(),
         input_preprocess=input_preprocess,
         target_preprocess=target_preprocess
      }
   )
end

function Bowl:loadClasses()
   local classes = {}
   for i=0, 120 do
      classes[i] = i
   end
   return classes
end

function Bowl:loadTrainValid()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   local data = self:loadData(self._train_file)
   if self._shuffle then
      print"shuffling train/valid set"
      data = data:index(1, torch.randperm(data:size(1)):long())
   end
   local size = math.floor(data:size(1)*(1-self._valid_ratio))
   local train_data = data:narrow(1, 1, size)
   self:setTrainSet(self:createDataSet(train_data, 'train'))
   local start = size + 1
   local size = data:size(1)-start
   local valid_data = data:narrow(1, start, size)
   self:setValidSet(self:createDataSet(valid_data, 'valid'))
end

function Bowl:loadTest()
   local test_data = self:loadData(self._test_file)
   self:setTestSet(self:createDataSet(test_data, 'test'))
end

function Bowl:createDataSet(data, which_set)
   local inputs = data:narrow(2, 1, self._feature_size):clone()
   inputs = inputs:type('torch.FloatTensor')
   inputs:resize(inputs:size(1), unpack(self._image_size))
   if self._scale then
      parent.rescale(inputs, self._scale[1], self._scale[2])
   end

   local targets = nil
   if which_set ~= 'test' then
      targets = data:select(2, self._feature_size+1):clone()
   else
      targets = torch.ByteTensor(data:size(1), 1):fill(1)
   end
   
   -- class 0 will have index 1, class 1 index 2, and so on.
   targets:add(1):resize(targets:size(1))
   targets = targets:type('torch.FloatTensor')
   print(inputs:size(), targets:size(), inputs:dim(), targets:dim())
   
   -- construct inputs and targets dp.Views 
   local input_v, target_v = dp.ImageView(), dp.ClassView()
   --self.targets = dp.ClassView()
   
   input_v:forward(self._image_axes, inputs)
   target_v:forward('b', targets)
   --self.targets:forward('b', targets)
 
   target_v:setClasses(self._classes)  -- Make sure that it is correct
   --self.targets:setClasses(self._classes)  -- Make sure that it is correct
   -- construct dataset
   return dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
end

function Bowl:loadData(file_name)
   local data = torch.load(self._data_path..file_name)
   data = data:narrow(1, 1, 5000):clone()
   return data
end
