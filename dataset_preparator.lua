require 'torch'


local function getPerm(n)
	local perm = torch.IntTensor(n)
	for i = n, 1, -1 do
		perm[i] = math.random(1, i)
	end
	return perm
end

local function shuffle(tensor, perm, fk)
	local n = tensor:size(1)
	assert(n == perm:size(1))
	-- local perm = perm or getPerm(n)
    for i = n, 1, -1 do
    	k = perm[i]
        -- tensor[i], tensor[k] = tensor[k], tensor[i]
        if not fk then
        	temp = tensor[i]:clone()
        else
        	temp = tensor[i]
        end
        tensor[i] = tensor[k]
        tensor[k] = temp

    end
end



local preparator = {}

function preparator.expandLabels(labels, n_classes)
	local n = labels:size(1)
	local targets = torch.Tensor(n, n_classes):zero()

	for i = 1, n do
		targets[i][labels[i]] = 1
	end
	return targets
end

function preparator.shuffleDataset(dataset)
	local perm = getPerm(dataset:size())
    shuffle(dataset.data, perm)
    shuffle(dataset.labels, perm, true)
end

function preparator.randomSplit(dataset, proportion)
	assert(proportion >= 0 and proportion <= 1)

	preparator.shuffleDataset(dataset)

	local n = dataset:size()
    local border = math.floor(n * proportion)
    local train = dataset:subset(1, border)
    local test = dataset:subset(border + 1, n - border)

    return train, test
end


function preparator.normalize(data, mean_, std_)
    local mean = mean_ or data:view(data:size(1), -1):mean(1)
    local std = std_ or data:view(data:size(1), -1):std(1, true)
    for i=1,data:size(1) do
        data[i]:add(-mean[1][i])
        if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
        end
    end
    return mean, std
  end

function preparator.normalizeGlobal(data, mean_, std_)
    local std = std_ or data:std()
    local mean = mean_ or data:mean()
    data:add(-mean)
    data:mul(1/std)
    return mean, std
end

function preparator.getLabelStatistics(labels)
	local stat = {}
	local weights = {}
	local mx = 0
	for i = 1, labels:size(1) do
		stat[labels[i]] = ( stat[labels[i]] or 0 ) + 1
	end
	for class, k in pairs(stat) do
		mx = math.max(mx, stat[class])
	end
	for class, k in pairs(stat) do
		weights[class] = mx / stat[class]
	end
	return stat, weights
end


return preparator