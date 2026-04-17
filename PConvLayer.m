classdef PConvLayer < nnet.layer.Layer & nnet.layer.Formattable
    %% Partial Convolution (PConv) Layer
    % Former Author: JoeyBG.
    % Improved By: JoeyBG.
    % Date: 2026-04-07.
    % Affiliate: Beijing Institute of Technology.
    % Platform: MATLAB R2025b.
    %
    % Introduction:
    %   This custom layer implements the Partial Convolution (PConv) module.
    %   It splits the input channels, applies a 3x3 convolution to a specific 
    %       fraction of the channels, and leaves the rest untouched to reduce 
    %       redundant computation and memory access.
    
    properties
        DimConv                                                             % Number of channels to apply convolution
        DimUntouched                                                        % Number of untouched channels
    end
    
    properties (Learnable)
        Weights                                                             % Convolution weights for the PConv operation
    end
    
    methods
        function layer = PConvLayer(dim, numDiv, NameValueArgs)
            % Construct the PConv Layer
            arguments
                dim
                numDiv = 4
                NameValueArgs.Name = ''
            end
            
            layer.Name = NameValueArgs.Name;
            layer.DimConv = dim / numDiv;
            layer.DimUntouched = dim - layer.DimConv;
            
            % Initialize weights using He initialization
            layer.Weights = randn([3, 3, layer.DimConv, layer.DimConv], 'single') * sqrt(2 / (9 * layer.DimConv));
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time
            
            % Split the input along the channel dimension
            X1 = X(:, :, 1:layer.DimConv, :);
            X2 = X(:, :, layer.DimConv+1:end, :);
            
            % Apply 3x3 spatial convolution to the split part X1
            biasZeros = zeros(layer.DimConv, 1, 'single');
            X1_conv = dlconv(X1, layer.Weights, biasZeros, 'Padding', 'same');
            
            % Concatenate the convolved channels with the untouched channels
            Z = cat(3, X1_conv, X2);
        end
    end
end