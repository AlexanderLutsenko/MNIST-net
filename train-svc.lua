os.setlocale('en_US.UTF-8')

require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'pl'
require 'LinearSVC'

local loader = require 'dataset_loader'
local preparator = require 'dataset_preparator'
local utils = require 'train_utils'


function getBatch(dataset, batchSize, t)
	local k = math.min(dataset:size() - (t - 1), batchSize)
	local batch = dataset:subset(t, k)
    return batch.data, preparator.expandLabels(batch.labels, 10)
    -- return batch.data, batch.labels
end


function train(model, dataset, batchSize, confusion)
	print('<trainer> on training set:')
	local parameters, gradParameters = model:getParameters()

	model:training()
	confusion:zero()
	local time = sys.clock()

   	preparator.shuffleDataset(dataset)

   	local n = dataset:size()
   	for t = 1, n, batchSize do
   		collectgarbage()
   		if (t - 1) % 1000 == 0 then
      		print(t - 1 .. '/' .. n)
      	end

   		local inputs, targets = getBatch(dataset, batchSize, t)

   		local eval_err = function(parameters)
   			-- model:zeroGradParameters()
		    -- local preds = model:forward(inputs)

		    -- criterion = nn.CrossEntropyCriterion()
		    -- err = criterion:forward(preds, targets)
		    -- t = criterion:backward(preds, targets)

      --   	-- model:backward(inputs, targets)
      --   	model:backward(inputs, t)

		    model:zeroGradParameters()
		    local preds = model:forward(inputs)
        	model:backward(inputs, targets)

        	for i = 1, inputs:size(1) do
         		confusion:add(preds[i], targets[i])
      		end	
      		return _, gradParameters
		end

 		optim.adadelta(eval_err, parameters, {}, {})
   end
   
   time = sys.clock() - time
   time = time / n
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
   print(confusion)

   return (confusion.totalValid * 100)
end


function test(model, dataset, batchSize, confusion)
	print('<trainer> on testing Set:')

	model:evaluate()
	confusion:zero()
   	local time = sys.clock()

   	local n = dataset:size()
   	for t = 1, n, batchSize do
      	local inputs, targets = getBatch(dataset, batchSize, t)
      	local preds = model:forward(inputs)
      	for i = 1, inputs:size(1) do
         	confusion:add(preds[i], targets[i])
      	end
   	end

   	time = sys.clock() - time
   	time = time / n
   	print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')
   	print(confusion)
  	return (confusion.totalValid * 100)
end


function make_model()
	local model = nn.Sequential()

	-- model:add(nn.SpatialConvolutionMM(1, 50, 3, 3))
	-- model:add(nn.SpatialBatchNormalization(50))
	-- model:add(nn.PReLU(50))

	-- model:add(nn.SpatialConvolutionMM(50, 50, 3, 3))
	-- model:add(nn.SpatialBatchNormalization(50))
	-- model:add(nn.PReLU(50))
	-- model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) 


	-- model:add(nn.SpatialConvolutionMM(50, 100, 3, 3))
	-- model:add(nn.SpatialBatchNormalization(100))
	-- model:add(nn.PReLU(100))

	-- model:add(nn.SpatialConvolutionMM(100, 100, 3, 3))
	-- model:add(nn.SpatialBatchNormalization(100))
	-- model:add(nn.PReLU(100))
	-- model:add(nn.SpatialMaxPooling(2, 2, 2, 2))


	-- model:add(nn.SpatialConvolutionMM(100, 150, 3, 3))
	-- model:add(nn.SpatialBatchNormalization(150))
	-- model:add(nn.PReLU(150))

	-- model:add(nn.SpatialConvolutionMM(150, 150, 3, 3))
	-- model:add(nn.SpatialBatchNormalization(150))
	-- model:add(nn.PReLU(150))

	-- model:add(nn.SpatialDropout(0.5))
	-- model:add(nn.Reshape(150*1*1, true))
	-- model:add(nn.LinearSVC(150*1*1, 10, 0.3))





	model:add(nn.SpatialConvolutionMM(1, 50, 3, 3))
	model:add(nn.SpatialBatchNormalization(50))
	model:add(nn.PReLU(50))

	model:add(nn.SpatialConvolutionMM(50, 50, 3, 3))
	model:add(nn.SpatialBatchNormalization(50))
	model:add(nn.PReLU(50))
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) 


	model:add(nn.SpatialConvolutionMM(50, 100, 3, 3))
	model:add(nn.SpatialBatchNormalization(100))
	model:add(nn.PReLU(100))

	model:add(nn.SpatialConvolutionMM(100, 100, 3, 3))
	model:add(nn.SpatialBatchNormalization(100))
	model:add(nn.PReLU(100))
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))


	model:add(nn.SpatialConvolutionMM(100, 150, 3, 3))
	model:add(nn.SpatialBatchNormalization(150))
	model:add(nn.PReLU(150))

	model:add(nn.SpatialConvolutionMM(150, 150, 3, 3))
	model:add(nn.SpatialBatchNormalization(150))
	model:add(nn.PReLU(150))

	model:add(nn.SpatialDropout(0.5))
	model:add(nn.Reshape(150*1*1, true))
	model:add(nn.LinearSVC(150*1*1, 10, 0.3))


	return model
end

function make_data()
	local train_archive = 'mnist.t7/train_32x32.t7'
	local test_archive = 'mnist.t7/test_32x32.t7'
	local classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	local trainData = loader.loadDatasetFromArchive(train_archive)
	local testData = loader.loadDatasetFromArchive(test_archive)

	-- trainData = trainData:subset(1, 6000)
	-- testData = testData:subset(1, 2000)

	trainData.data = trainData.data:float()
	testData.data = testData.data:float()
	preparator.normalizeGlobal(trainData.data)
	preparator.normalizeGlobal(testData.data)

	local confusion = optim.ConfusionMatrix(classes)
	return trainData, testData, confusion
end

function doIt()
	local threads = 2
	torch.manualSeed(1234)
	math.randomseed(1234)
	torch.setnumthreads(threads)
	torch.setdefaulttensortype('torch.FloatTensor')

	local save_folder = 'save'
	local logAndPlot = utils.getLogger(save_folder, 'logger.log', 'plot.pdf')

	local trainData, testData, confusion = make_data()
	-- local _, classWeights = preparator.getLabelStatistics(trainData.labels)

	local model = make_model()
	-- local model = torch.load(paths.concat(save_folder, 'net'))
	local batchSize = 10
	local epochs = 30


	best_acc = 0
	for epoch = 1, epochs do
	   	print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')

	   	local train_acc = train(model, trainData, batchSize, confusion)
	   	local test_acc = test(model, testData, 100, confusion)
	    logAndPlot(train_acc, test_acc)

	    -- save current net
	    if test_acc > best_acc then
	    	best_acc = test_acc
	    	utils.saveModel(model, save_folder, 'net')
	    end
	end
end

doIt()