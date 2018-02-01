using CNTK;
using System;
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
            // Same goes for the dimensions.
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
        
        static CNTKDictionary DefaultInitializer(uint seed)
        {
            return CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, seed);
        }
    }
}
