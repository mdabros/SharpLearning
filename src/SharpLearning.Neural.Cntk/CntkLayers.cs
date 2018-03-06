using System;
using System.Linq;
using CNTK;

namespace SharpLearning.Neural.Cntk
{
    /// <summary>
    /// Layers operations for CNTK
    /// </summary>
    public static class CntkLayers
    {
        public static DeviceDescriptor Device = DeviceDescriptor.UseDefaultDevice();

        public static Function Input(params int[] inputDim)
        {
            return Variable.InputVariable(inputDim, DataType.Float);
        }

        public static Function SoftMax(Variable input)
        {
            return CNTKLib.Softmax(input);
        }

        public static Function Dense(Variable input, int units, uint seed = 32, string outputName = "")
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
                    CNTKLib.SentinelValueForInferParamInitRank, seed),
                Device, "timesParam");

            var timesFunction = CNTKLib.Times(weights, input, "times");

            int[] s2 = { units };
            var bias = new Parameter(s2, 0.0f, Device, "plusParam");

            return CNTKLib.Times(weights, input) + bias;
        }

        public static Function Conv2D(Variable input, int filterW, int filterH, int filterCount,
             int strideW = 1, int strideH = 1, uint seed = 34, string outputName = "")
        {
            if (input.Shape.Rank != 3)
            {
                throw new ArgumentException("Conv2D layer requires shape rank 3, got rank " + input.Shape.Rank);
            }

            var inputChannels = input.Shape[2];

            var convWScale = 0.26;
            var convParams = new Parameter(new int[] { filterW, filterH, inputChannels, filterCount }, DataType.Float,
                CNTKLib.GlorotUniformInitializer(convWScale, -1, 2, seed), Device);

            return CNTKLib.Convolution(convParams, input, new int[] { strideW, strideH, inputChannels });
        }

        public static Function Pool2D(Variable input, int poolW, int poolH,
            PoolingType poolingType = PoolingType.Max,
            int strideW = 2, int strideH = 2, string outputName = "")
        {
            if (input.Shape.Rank != 3)
            {
                throw new ArgumentException("Pool2D layer requires shape rank 3, got rank " + input.Shape.Rank);
            }

            return CNTKLib.Pooling(input, PoolingType.Max,
                new int[] { poolW, poolH }, new int[] { strideW, strideH });
        }

        public static Function Dropout(Variable input, double dropoutRate, uint seed = 465)
        {
            return CNTKLib.Dropout(input, dropoutRate, seed);
        }

        public static Function BatchNormalizationLayer(Variable input, bool spatial,
            double initialScaleValue = 1, double initialBiasValue = 0, int bnTimeConst = 5000)
        {
            var biasParams = new Parameter(new int[] { NDShape.InferredDimension }, (float)initialBiasValue, Device, "");
            var scaleParams = new Parameter(new int[] { NDShape.InferredDimension }, (float)initialScaleValue, Device, "");
            var runningMean = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, Device);
            var runningInvStd = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, Device);
            var runningCount = Constant.Scalar(0.0f, Device);

            return CNTKLib.BatchNormalization(input, scaleParams, biasParams, runningMean, runningInvStd, runningCount,
                spatial, (double)bnTimeConst, 0.0, 1e-5 /* epsilon */);
        }
    }
}
