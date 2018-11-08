using System;
using System.Linq;
using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        /// <summary>
        /// Based on Dense from: https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/layers.py
        /// </summary>
        public static Function Dense(this Function input,
            int units,
            CNTKDictionary weightInitializer,
            CNTKDictionary biasInitializer,
            DeviceDescriptor device,
            DataType dataType,
            int inputRank = 0,
            int mapRank = 0)
        {
            if (inputRank != 0 && mapRank != 0)
            {
                throw new ArgumentException("Dense: inputRank and mapRank cannot be specified at the same time.");
            }

            var outputShape = NDShape.CreateNDShape(new int[] { units });
            var outputRank = outputShape.Dimensions.Count;

            var inputRanks = (inputRank != 0) ? inputRank : 1;
            var dimensions = Enumerable.Range(0, inputRanks)
                .Select(v => NDShape.InferredDimension).ToArray(); // infer all dimensions.
            var inputShape = NDShape.CreateNDShape(dimensions);

            int inferInputRankToMap;

            if (inputRank != 0)
            {
                inferInputRankToMap = -1; // means map_rank is not specified; input_rank rules.
            }
            else if (mapRank == 0)
            {
                inferInputRankToMap = 0;  // neither given: default to 'infer W to use all input dims'.
            }
            else
            {
                inferInputRankToMap = mapRank;  // infer W to use all input dims except the first static 'map_rank' ones.
            }

            var weightsDimensions = outputShape.Dimensions.ToList();
            weightsDimensions.AddRange(inputShape.Dimensions);
            var weightsShape = NDShape.CreateNDShape(weightsDimensions);

            var weights = new Parameter(weightsShape, dataType, weightInitializer, device, "w");

            // Weights and input is in reversed order compared to the original python code.
            // Same goes for the dimensions. This is because the python api reverses the dimensions internally.
            // The python API was made in this way to be similar to other deep learning toolkits. 
            // The C# and the C++ share the same column major layout.
            var r = CNTKLib.Times(weights, input, (uint)outputRank, inferInputRankToMap);

            if (biasInitializer != null)
            {
                var biasParameter = new Parameter(outputShape, dataType, biasInitializer, device, "b");
                r = r + biasParameter;
            }

            return r;
        }
    }
}
