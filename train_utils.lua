local utils = {}

function utils.getLogger(folder, logger_name, plot_name)
	local logger_path = paths.concat(folder, logger_name)
	local plot_path = paths.concat(folder, plot_name)

	local logger = optim.Logger(logger_path)
	local train_name = 'training'
	local test_name = 'testing'

	function logAndPlot(train_acc, test_acc)
		logger:add{[train_name] = train_acc, [test_name] = test_acc}

		gnuplot.title('Mean class accuracy')
		gnuplot.grid(true)
		gnuplot.xlabel('# of epoch')
		gnuplot.ylabel('accuracy, %')

		gnuplot.plot({train_name, torch.FloatTensor(logger.symbols[train_name]), '-'}, 
					 {test_name, torch.FloatTensor(logger.symbols[test_name]), '-'})

		gnuplot.axis('auto')
		gnuplot.figprint(plot_path)
	end
	return logAndPlot
end

function utils.saveModel(model, folder, name)
	local filename = paths.concat(folder, name)
	-- if paths.filep(filename) then
	--     os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
	-- end
	print('<utils> saving network to '.. filename)
	torch.save(filename, model)
end

return utils
