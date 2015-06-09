
local LinearSVC, parent = torch.class('nn.LinearSVC', 'nn.Module')

function LinearSVC:__init(inputSize, outputSize, c, autoscale)
    parent.__init(self)

    self.weight = torch.Tensor(outputSize, inputSize + 1)
    self.gradWeight = torch.Tensor(self.weight:size())
    self.c = c
    self.autoscale = autoscale or true
    self:reset()
end

function LinearSVC:reset(stdv)
    if stdv then
        stdv = stdv * math.sqrt(3)
    else
        stdv = 1./math.sqrt(self.weight:size(2))
    end
    if nn.oldSeed then
        for i=1,self.weight:size(1) do
            self.weight:select(1, i):apply(function()
                return torch.uniform(-stdv, stdv)
            end)
        end
    else
        self.weight:uniform(-stdv, stdv)
    end
    return self
end

function LinearSVC:_max_response(output)
    local mx = torch.max(output, 2)
    local mx = torch.repeatTensor(mx, 1, output:size(2))
    local function fmx(el1, el2)
        if el1 == el2 then return 1
        else return 0
        end
    end
    mx:map(output, fmx)
    return mx
end

function LinearSVC:_shift_output(output)
    return output * 2 - 1
end

function LinearSVC:_to_matrix(tensor)
    if tensor:dim() == 1 then
        tensor = tensor:view(1, tensor:size(1))
    elseif tensor:dim() ~= 2 then
        error('vector or batch of vectors (matrix) expected')
    end
    return tensor
end

function LinearSVC:updateOutput(input)
    self.input = self:_to_matrix(input)

    local bias = torch.Tensor(input:size(1), 1):fill(1)
    self.input = torch.cat(input, bias, 2)

    self.output = self.input * self.weight:t()
    self.output = self:_max_response(self.output)
    return self.output
end

function LinearSVC:backward(_, gradOutput, scale)
    gradOutput = self:_to_matrix(gradOutput)
    gradOutput = self:_shift_output(gradOutput)

    local subGrad = self:_calc_subgrad(_, gradOutput)

    self:updateGradInput(input, gradOutput, subGrad)
    self:accGradParameters(input, gradOutput, scale, subGrad)
    return self.gradInput
end

function LinearSVC:_calc_subgrad(_, gradOutput)
    --    X*W^t
    local subGrad = self.input * self.weight:t()
    --    1 - X*W^t(.)T
    subGrad:cmul(gradOutput)
    subGrad:mul(-1)
    subGrad:add(1)
    --    max( 1 - X*W^t(.)T , 0 )
    subGrad[torch.lt(subGrad, 0)] = 0
    --    T (.) max( 1 - X*W^t(.)T , 0 )
    subGrad:cmul(gradOutput)
    --    - 2c/n * T (.) max( 1 - X*W^t(.)T , 0 )
    local batch_size = self.input:size(1)
    local mult_const = -2*self.c / batch_size
    subGrad:mul(mult_const)
    return subGrad
end

-- function LinearSVC:_summed_outer_product(t1, t2)
--     local n_summed = t1:size(2)
--     local outer = torch.Tensor(t1:size(1), t2:size(2)):zero()
--     for i = 1, n_summed do
--         outer:addr(t1:select(2, i), t2[i])
--     end
--     return outer
-- end

function LinearSVC:_summed_outer_product(t1, t2)
    local n_summed = t1:size(1)
    local outer = torch.Tensor(t1:size(2), t2:size(2)):zero()
    for i = 1, n_summed do
        outer:addr(t1[i], t2[i])
    end
    -- outer:div(n_summed)
    return outer
end

function LinearSVC:updateGradInput(_, gradOutput, subGrad)
    --   - 2c/n * T (.) max( 1 - X*W^t(.)T , 0 )
    local subGrad = subGrad or self:_calc_subgrad(_, gradOutput)
    --   - W | [2c/n * T (.) max( 1 - X*W^t(.)T , 0 )]
    local grad = self:_summed_outer_product(subGrad:t(), self.weight)
    --   don't forget to cut off bias
    self.gradInput = grad:narrow(2, 1, grad:size(2) - 1)
    return self.gradInput
end

function LinearSVC:accGradParameters(_, gradOutput, scale, subGrad)
    -- Do not update parameters
end

-- function LinearSVC:accGradParameters(_, gradOutput, scale, subGrad)
--     --   - 2c/n * T (.) max( 1 - X*W^t(.)T , 0 )
--     local subGrad = subGrad or self:_calc_subgrad(_, gradOutput)
--     --   - X | [2c/n * T (.) max( 1 - X*W^t(.)T , 0 )]
--     local grad = self:_summed_outer_product(subGrad, self.input)
--     --   W - X | [2c/n * T (.) max( 1 - X*W^t(.)T , 0 )]
--     grad:add(self.weight)
--     --  don't touch bias!
--     grad:t()[-1]:add(-self.weight:t()[-1])

--     self.gradWeight:zero()
--     self.gradWeight:add(grad)
    
--     if self.autoscale then
--         local inputNorm = self.gradInput:sum(1):norm()
--         local weightNorm = self.gradWeight:sum(1):norm()
--         scale = inputNorm / weightNorm
--     end
--     self.gradWeight:mul(scale)
-- end

-- we do not need to accumulate parameters when sharing
LinearSVC.sharedAccUpdateGradParameters = LinearSVC.accUpdateGradParameters

function LinearSVC:__tostring__()
    return torch.type(self) ..
        string.format('(%d -> %d)', self.weight:size(2) - 1, self.weight:size(1))
end