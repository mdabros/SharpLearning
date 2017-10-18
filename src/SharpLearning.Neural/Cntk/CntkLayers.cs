using System;
using System.Linq;
using CNTK;

namespace SharpLearning.Neural.Cntk
{
    public enum Activation
    {
        None,
        ReLU,
        Sigmoid,
        Tanh
    }

    /// <summary>
    /// Temporary class to help create layers via cntk.
    /// </summary>
    public static class CntkLayers
    {
        public static DeviceDescriptor Device = DeviceDescriptor.UseDefaultDevice();

        public static Function Input(params int[] inputDim)
        {
            return Variable.InputVariable(inputDim, DataType.Float);
        }

        public static Function Dense(Variable input, int units, string outputName = "")
        {
            if (input.Shape.Rank != 1)
            {
                // 
                int newDim = input.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                input = CNTKLib.Reshape(input, new int[] { newDim });
            }

            int inputDim = input.Shape[0];

            int[] s = { units, inputDim };
            var weights = new Parameter((NDShape)s, DataType.Float,
                CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1),
                Device, "timesParam");

            var timesFunction = CNTKLib.Times(weights, input, "times");

            int[] s2 = { units };
            var bias = new Parameter(s2, 0.0f, Device, "plusParam");

            return CNTKLib.Times(weights, input) + bias;
        }

        public static Function Activation(Variable input, Activation activation)
        {
            switch (activation)
            {
                default:
                case Cntk.Activation.None:
                    return input;
                case Cntk.Activation.ReLU:
                    return CNTKLib.ReLU(input);
                case Cntk.Activation.Sigmoid:
                    return CNTKLib.Sigmoid(input);
                case Cntk.Activation.Tanh:
                    return CNTKLib.Tanh(input);
            }
        }

        public static Function Conv2D(Variable input, int filterW, int filterH, int filterCount,
             int strideW = 1, int strideH = 1, string outputName = "")
        {
            if (input.Shape.Rank != 3)
            {
                throw new ArgumentException($"Conv2D layer requires shape rank 3, got rank {input.Shape.Rank}");
            }

            var inputChannels = input.Shape[2];

            var convWScale = 0.26;
            var convParams = new Parameter(new int[] { filterW, filterH, inputChannels, filterCount }, DataType.Float,
                CNTKLib.GlorotUniformInitializer(convWScale, -1, 2), Device);

            return CNTKLib.Convolution(convParams, input, new int[] { strideW, strideH, inputChannels });
        }

        public static Function Pool2D(Variable input, int poolW, int poolH, 
            PoolingType poolingType = PoolingType.Max,
            int strideW = 1, int strideH = 1, string outputName = "")
        {
            if (input.Shape.Rank != 3)
            {
                throw new ArgumentException($"Pool2D layer requires shape rank 3, got rank {input.Shape.Rank}");
            }

            return CNTKLib.Pooling(input, PoolingType.Max,
                new int[] { poolW, poolH}, new int[] { strideW, strideH});
        }

        public static Function Dropout(Variable input, double dropoutRate, uint seed)
        {
            return CNTKLib.Dropout(input, dropoutRate, seed);
        }
    }
}
