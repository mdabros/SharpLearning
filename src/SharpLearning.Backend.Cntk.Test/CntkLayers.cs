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
        /// </summary>
        /// <param name="x"></param>
        /// <param name="shape"></param>
        /// <param name="activation"></param>
        /// <param name="init"></param>
        /// <param name="inputRank"></param>
        /// <param name="mapRank"></param>
        /// <param name="bias"></param>
        /// <param name="initBias"></param>
        /// <returns></returns>
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

            init = init ?? m_getDefaultInitializer();
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
        
        static CNTKDictionary DefaultInitializer(uint seed)
        {
            return CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, seed);
        }
    }
}
