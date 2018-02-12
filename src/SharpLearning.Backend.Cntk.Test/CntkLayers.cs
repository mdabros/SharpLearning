using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;

// To avoid name-clash with SharpLearning.Backend.DataType.
using CntkDataType = CNTK.DataType;

namespace SharpLearning.Backend.Cntk.Test
{
    /// <summary>
    /// Helper class based on: https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/layers.py
    /// 
    /// Note that this class is not meant to be part of the backend project.
    /// The implementation of each method should help illustrate 
    /// which base operators and concept are necesarry to support in the low-level api of the backend project.
    /// </summary>
    public class CntkLayers
    {
        readonly DeviceDescriptor m_device;
        readonly CntkDataType m_dataType;
        readonly Func<CNTKDictionary> m_getDefaultInitializer;

        public CntkLayers()
            : this(DeviceDescriptor.UseDefaultDevice(), CntkDataType.Float, 
                    () => DefaultInitializer(seed: 42))
        {
        }

        public CntkLayers(DeviceDescriptor device, CntkDataType dataType)
            : this(device, dataType, () => DefaultInitializer(seed: 42))
        {
        }

        public CntkLayers(DeviceDescriptor device, CntkDataType dataType, 
            Func<CNTKDictionary> getDefaultInitializer)
        {
            if (device == null) { throw new ArgumentNullException("device"); }
            if (getDefaultInitializer == null) { throw new ArgumentNullException("getDefaultInitializer"); }

            m_device = device;
            m_dataType = dataType;
            m_getDefaultInitializer = getDefaultInitializer;
        }

        /// <summary>
        /// From Dense in: https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/layers.py
        /// 
        /// Input description is from python, so might not match this implementation completely. 
        /// But is included to provide the itension of each input.
        /// 
        /// </summary>
        /// <param name="x">Input to the dense operation</param>
        /// <param name="shape">Shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer</param>
        /// <param name="activation">Activation (:class:`~cntk.ops.functions.Function`, defaults to identity): optional function to apply at the end, e.g. `relu`</param>
        /// <param name="init">init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`</param>
        /// <param name="inputRank">input_rank (int, defaults to `None`): number of inferred axes to add to W (`map_rank` must not be given)</param>
        /// <param name="mapRank"> map_rank (int, defaults to `None`): expand W to leave exactly `map_rank` axes (`input_rank` must not be given)</param>
        /// <param name="bias">bias (bool, optional, defaults to `True`): the layer will have no bias if `False` is passed here</param>
        /// <param name="initBias">init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`</param>
        /// <returns>cntk.ops.functions.Function: A function that accepts one argument and applies the operation to it</returns>
        public Function Dense(Variable x, int shape, Func<Variable, Function> activation = null, CNTKDictionary init = null,
            int inputRank = 0, int mapRank = 0, bool bias = true, CNTKDictionary initBias = null)
        {
            if(inputRank != 0 && mapRank != 0)
            {
                throw new ArgumentException("Dense: inputRank and mapRank cannot be specified at the same time.");
            }

            var outputShape = NDShape.CreateNDShape(new int[] { shape });
            var outputRank = outputShape.Dimensions.Count;

            var inputRanks = (inputRank != 0) ? inputRank : 1;
            var dimensions = Enumerable.Range(0, inputRanks).Select(v => NDShape.InferredDimension).ToArray(); // infer all dimensions.
            var inputShape = NDShape.CreateNDShape(dimensions);
            
            int inferInputRankToMap;

            if (inputRank != 0)
                inferInputRankToMap = -1; // means map_rank is not specified; input_rank rules.
            else if (mapRank == 0)
                inferInputRankToMap = 0;  // neither given: default to 'infer W to use all input dims'.
            else
                inferInputRankToMap = mapRank;  // infer W to use all input dims except the first static 'map_rank' ones.

            var wDimensions = outputShape.Dimensions.ToList();
            wDimensions.AddRange(inputShape.Dimensions);
            var wShape = NDShape.CreateNDShape(wDimensions);

            init = init ?? m_getDefaultInitializer(); // should also be possible to provide a fixed value.
            var w = new Parameter(wShape, m_dataType, init, m_device, "w");

            // Weights and input is in reversed order compared to the original python code.
            // Same goes for the dimensions. This is because the python api reverses the dimensions internally.
            // The python API was made in this way to be similar to other deep learning toolkits. 
            // The C# and the C++ share the same column major layout.
            var r = CNTKLib.Times(w, x, (uint)outputRank, inferInputRankToMap);

            if(bias)
            {
                initBias = initBias ?? m_getDefaultInitializer(); // should also be possible to provide a fixed value.
                var b = new Parameter(outputShape, m_dataType, initBias, m_device, "b");
                r = r + b;
            }

            if(activation != null)
            {
                r = activation(r);
            }

            return r;
        }

        /// <summary>
        /// From Dropout in: https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/layers.py
        /// 
        /// Input description is from python, so might not match this implementation completely. 
        /// But is included to provide the itension of each input.
        /// </summary>
        /// <param name="x">Input to the dropout operation</param>
        /// <param name="dropOutRate">Probability of dropping out an element, mutually exclusive with ``keep_prob``</param>
        /// <param name="keepProb">Probability of keeping an element, mutually exclusive with ``dropout_rate``</param>
        /// <param name="seed">random seed</param>
        /// <returns>cntk.ops.functions.Function: A function that accepts one argument and applies the operation to it</returns>
        public Function Dropout(Variable x, double dropOutRate, double keepProb = 0.0, uint seed = 34)
        {
            if (dropOutRate == 0.0 && keepProb == 0.0)
                throw new ArgumentException("Dropout: either dropout_rate or keep_prob must be specified.");
            else if (dropOutRate != 0.0 && keepProb != 0.0)
                throw new ArgumentException("Dropout: dropout_rate and keep_prob cannot be specified at the same time.");
            else if (keepProb != 0.0)
            {
                if (keepProb < 0.0 || keepProb >= 1.0)
                    throw new ArgumentException("Dropout: keep_prob must be in the interval [0,1]");

                dropOutRate = 1 - keepProb;
            }

            return CNTKLib.Dropout(x, dropOutRate, seed);
        }

        /// <summary>
        /// From Convolution2D in: https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/layers.py
        /// Simplified compared to python version.
        /// 
        /// Input description is from python, so might not match this implementation completely. 
        /// But is included to provide the itension of each input.
        /// </summary>
        /// <param name="x">Input to the dense operation</param>
        /// <param name="filterShape">filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.</param>
        /// <param name="numFilters">num_filters (int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).</param>
        /// <param name="activation"> activation (:class:`~cntk.ops.functions.Function`, defaults to `identity`): optional function to apply at the end, e.g. `relu`</param>
        /// <param name="init">init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`</param>
        /// <param name="pad"> pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the filter will be shifted over the "valid"
        /// area of input, that is, no value outside the area is used.If ``pad=True`` on the other hand,
        /// the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
        /// Use a `tuple` to specify a per-axis value.</param>
        /// <param name="strides">strides (`int` or `tuple` of `ints`, defaults to 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.</param>
        /// <param name="bias"> bias (bool, defaults to `True`): the layer will have no bias if `False` is passed here</param>
        /// <param name="initBias"> init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`</param>
        /// <param name="reductionRank">reduction_rank (`int`, defaults to 1): set to 0 if input items are scalars (input has no depth axis), e.g. an audio signal or a black-and-white image</param>
        /// <param name="dilation">dilation (tuple, optional): the dilation value along each axis, default 1 mean no dilation.</param>
        /// <param name="groups">groups (`int`, default 1): number of groups during convolution, that controls the connections between input and output channels. Deafult value is 1, 
        /// which means that all input channels are convolved to produce all output channels.A value of N would mean that the input(and output) channels are
        /// divided into N groups with the input channels in one group(say i-th input group) contributing to output channels in only one group(i-th output group).
        /// Number of input and output channels must be divisble by value of groups argument.Also, value of this argument must be strictly positive, i.e.groups > 0.</param>
        /// <returns></returns>
        public Function Convolution2D(Variable x, NDShape filterShape, int numFilters,
            Func<Variable, Function> activation = null,
            CNTKDictionary init = null,
            bool pad = false,
            NDShape strides = null,
            bool bias = true,
            CNTKDictionary initBias = null,
            uint reductionRank = 1,
            int dilation = 1,
            uint groups = 1)
        {
            if (filterShape.Dimensions.Count != 2)
            { throw new ArgumentException($"Convolution2D requires a 2D filter, e.g. (3,3). Got {filterShape.Dimensions.Count}D."); }

            return Convolution(x: x,
                filterShape: filterShape,
                numFilters: numFilters,
                activation: activation,
                init: init,
                pad: pad,
                strides: strides,
                sharing: true,
                bias: bias,
                initBias: initBias,
                reductionRank: reductionRank,
                dilation: dilation,
                groups: groups);
        }

        /// <summary>
        /// From Convolution in: https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/layers.py
        /// 
        /// Input description is from python, so might not match this implementation completely. 
        /// But is included to provide the itension of each input.
        /// </summary>
        /// <param name="x">Input to the dense operation</param>
        /// <param name="filterShape">(`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.</param>
        /// <param name="numFilters">(int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).</param>
        /// <param name="sequential">(bool, defaults to `False`): if `True`, also convolve along the dynamic axis. ``filter_shape[0]`` corresponds to dynamic axis.</param>
        /// <param name="activation">(:class:`~cntk.ops.functions.Function`, defaults to `identity`): optional function to apply at the end, e.g. `relu`</param>
        /// <param name="init">(scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`</param>
        /// <param name="pad">(`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the filter will be shifted over the "valid"
        /// area of input, that is, no value outside the area is used.If ``pad=True`` on the other hand,
        /// the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
        /// Use a `tuple` to specify a per-axis value.</param>
        /// <param name="strides">(`int` or `tuple` of `ints`, defaults to 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.</param>
        /// <param name="sharing">sharing (bool, defaults to `True`): When `True`, every position uses the same Convolution kernel.  When `False`, you can have a different Convolution kernel per position, but `False` is not supported.</param>
        /// <param name="bias">bias (bool, optional, defaults to `True`): the layer will have no bias if `False` is passed here</param>
        /// <param name="initBias">init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`</param>
        /// <param name="reductionRank">(`int`, defaults to 1): set to 0 if input items are scalars (input has no depth axis), e.g. an audio signal or a black-and-white image
        /// that is stored with tensor shape(H, W) instead of(1, H, W)</param>
        /// <param name="transposeWeights">(bool, defaults to `False`): When this is `True` this is convolution, otherwise this is correlation (which is common for most toolkits)</param>
        /// <param name="dilation">(tuple, optional): the dilation value along each axis, default 1 mean no dilation.</param>
        /// <param name="groups">(`int`, default 1): number of groups during convolution, that controls the connections between input and output channels. Deafult value is 1, 
        /// which means that all input channels are convolved to produce all output channels.A value of N would mean that the input(and output) channels are
        /// divided into N groups with the input channels in one group(say i-th input group) contributing to output channels in only one group(i-th output group).
        /// Number of input and output channels must be divisble by value of groups argument.Also, value of this argument must be strictly positive, i.e.groups > 0.</param>
        /// <param name="maxTempMemSizeInSamples"> max_temp_mem_size_in_samples (int, defaults to 0): Limits the amount of memory for intermediate convolution results.  A value of 0 means, memory is automatically managed.</param>
        /// <returns>cntk.ops.functions.Function: A function that accepts one argument and applies the convolution operation to it</returns>
        Function Convolution(Variable x, NDShape filterShape, int numFilters,
            bool sequential = false,
            Func<Variable, Function> activation = null,
            CNTKDictionary init = null,
            bool pad = false,
            NDShape strides = null,
            bool sharing = true, // must be true currently
            bool bias = true,
            CNTKDictionary initBias = null,
            uint reductionRank = 1,
            bool transposeWeights = false, 
            int dilation = 1,
            uint groups = 1,
            int maxTempMemSizeInSamples = 0)
        {
            if (pad)
                throw new ArgumentException("Padding not supported"); 
            if (reductionRank != 0  && reductionRank != 1)
                throw new ArgumentException("Convolution: reduction_rank must be 0 or 1");
            if (transposeWeights)
                throw new ArgumentException("Convolution: transpose_weight option currently not supported");
            if (!sharing)
                throw new ArgumentException("Convolution: sharing option currently must be True");
            if (groups <= 0)
                throw new ArgumentException("Convolution: groups must be strictly positive, i.e. groups > 0.");
            if (groups > 1)
                throw new ArgumentException("Convolution: groups > 1, is not currently supported by Convolution layer. For group convolution with groups > 1, use CNTK's low-level convolution node (cntk.convolution).");

            var emulatingOutputDepth = numFilters == 0; // if no filters specified.
            var emulatingInputDepth = reductionRank == 0;

            var actualOutputChannelsShape = emulatingOutputDepth ? 1 : numFilters; // might need to be a shape/dimensions array.              
            var actualReductionShape = NDShape.InferredDimension;
            var actualFilterShape = filterShape;
            
            // add the dimension to the options as well
            var numEmulatedAxes = emulatingInputDepth;
            var sharings = new List<bool> { sharing };
            var pads = new List<bool> { pad };

            if (numEmulatedAxes)
            {
                strides.Dimensions.Add(1); // strides = (1,) * num_emulated_axes + strides
                sharings.Add(true); // sharing = (True,) * num_emulated_axes + sharing
                pads.Add(false); // pad = (False,) * num_emulated_axes + pad
            }

            var kernel_shape = actualFilterShape.Dimensions.ToList();
            kernel_shape.Add(actualReductionShape); // kernel := filter plus reductionDims;

            var fShape = actualFilterShape.Dimensions.ToList();
            var wShape = kernel_shape.ToList();
            fShape.ForEach(d => wShape.Add(d));

            init = init ?? m_getDefaultInitializer(); // should also be possible to provide a fixed value.
            var w = new Parameter(wShape.ToArray(), m_dataType, init, m_device, "w");

            // Weights and input is in reversed order compared to the original python code.
            // Same goes for the dimensions. This is because the python api reverses the dimensions internally.
            // The python API was made in this way to be similar to other deep learning toolkits. 
            // The C# and the C++ share the same column major layout.
            var r = CNTKLib.Convolution(w, x, 
                strides, 
                new BoolVector(sharings),
                new BoolVector(pads),
                new int[] { dilation },
                reductionRank,
                groups);

            if (bias)
            {
                initBias = initBias ?? m_getDefaultInitializer(); // should also be possible to provide a fixed value.
                var b = new Parameter(r.Output.Shape, m_dataType, initBias, m_device, "b");
                r = r + b;
            }

            if (activation != null)
            {
                r = activation(r);
            }

            return r;
        }
        
        static CNTKDictionary DefaultInitializer(uint seed)
        {
            return CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, seed);
        }
    }
}
