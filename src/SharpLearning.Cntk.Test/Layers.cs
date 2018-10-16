using System;
using System.Linq;
using CNTK;

namespace SharpLearning.Cntk.Test
{
    public static class Layers
    {
        public static Function ReLU(this Function x)
        {
            return CNTKLib.ReLU(x);
        }

        public static Function Sigmoid(this Function x)
        {
            return CNTKLib.Sigmoid(x);
        }

        public static Function Softmax(this Function x)
        {
            return CNTKLib.Softmax(x);
        }

        public static Function Input(NDShape inputShape, DataType d)
        {
            return Variable.InputVariable(inputShape, d);
        }

        /// <summary>
        /// Based on Dense from: https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/layers.py
        /// </summary>
        /// <param name="x"></param>
        /// <param name="units"></param>
        /// <param name="weightInitializer"></param>
        /// <param name="biasInitializer"></param>
        /// <param name="bias"></param>
        /// <param name="inputRank"></param>
        /// <param name="mapRank"></param>
        /// <returns></returns>
        public static Function Dense(this Function x, int units,
            DataType d, DeviceDescriptor device,
            Initializer weightInitializer = Initializer.GlorotUniform,
            Initializer biasInitializer = Initializer.Zeros,
            bool bias = true, int inputRank = 0, int mapRank = 0)
        {
            return Dense(x, units,
                d, device,
                Initializers.Create(weightInitializer),
                Initializers.Create(biasInitializer),
                bias, inputRank, mapRank);
        }

        /// <summary>
        /// Based on Dense from: https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/layers.py
        /// </summary>
        /// <param name="x"></param>
        /// <param name="units"></param>
        /// <param name="weightInitializer"></param>
        /// <param name="biasInitializer"></param>
        /// <param name="bias"></param>
        /// <param name="inputRank"></param>
        /// <param name="mapRank"></param>
        /// <returns></returns>
        public static Function Dense(this Function x, int units,
            DataType d, DeviceDescriptor device, CNTKDictionary weightInitializer,
            CNTKDictionary biasInitializer, bool bias = true, int inputRank = 0, int mapRank = 0)
        {
            if (inputRank != 0 && mapRank != 0)
            {
                throw new ArgumentException("Dense: inputRank and mapRank cannot be specified at the same time.");
            }

            var outputShape = NDShape.CreateNDShape(new int[] { units });
            var outputRank = outputShape.Dimensions.Count;

            var inputRanks = (inputRank != 0) ? inputRank : 1;
            var dimensions = Enumerable.Range(0, inputRanks).Select(v => NDShape.InferredDimension).ToArray(); // infer all dimensions.
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

            var wDimensions = outputShape.Dimensions.ToList();
            wDimensions.AddRange(inputShape.Dimensions);
            var wShape = NDShape.CreateNDShape(wDimensions);

            var w = new Parameter(wShape, d, weightInitializer, device, "w");

            // Weights and input is in reversed order compared to the original python code.
            // Same goes for the dimensions. This is because the python api reverses the dimensions internally.
            // The python API was made in this way to be similar to other deep learning toolkits. 
            // The C# and the C++ share the same column major layout.
            var r = CNTKLib.Times(w, x, (uint)outputRank, inferInputRankToMap);

            if (bias)
            {
                var b = new Parameter(outputShape, d, biasInitializer, device, "b");
                r = r + b;
            }

            return r;
        }
    }
}
