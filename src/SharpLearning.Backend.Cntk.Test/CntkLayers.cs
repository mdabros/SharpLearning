using CNTK;
using System;
using System.Linq;

// To avoid name-clash with SharpLearning.Backend.DataType.
using CntkDataType = CNTK.DataType;

namespace SharpLearning.Backend.Cntk.Test
{
    /// <summary>
    /// Helper class to make CNTK operator creation more simple.
    /// </summary>
    public static class CntkLayers
    {
        public static DeviceDescriptor Device = DeviceDescriptor.UseDefaultDevice();
        public static CntkDataType DataType = CntkDataType.Float;

        public static Function Input(params int[] inputDim)
        {
            return Variable.InputVariable(inputDim, DataType);
        }

        public static Function SoftMax(Variable input)
        {
            return CNTKLib.Softmax(input);
        }

        public static Function Dense(Variable input, int units, uint seed = 32, string outputName = "")
        {
            if (input.Shape.Rank != 1)
            {
                // Flatten dimensions.
                var newDim = input.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                input = CNTKLib.Reshape(input, new int[] { newDim });
            }

            // Use GlorotUniform for weight initialization.
            var initializer = CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, seed);

            var inputDim = input.Shape[0];

            var weights = new Parameter(new int[] { units, inputDim },
                DataType, initializer, Device, "timesParam");

            // Bias is initialized to 0.0.
            var bias = new Parameter(new int[] { units }, 0.0f, Device, "plusParam");

            return CNTKLib.Times(weights, input) + bias;
        }

        public static Function ActivationFunction(Variable input, Activation activation)
        {
            switch (activation)
            {
                default:
                case Activation.None:
                    return input;
                case Activation.ReLU:
                    return CNTKLib.ReLU(input);
                case Activation.LeakyReLU:
                    return CNTKLib.LeakyReLU(input);
                case Activation.Sigmoid:
                    return CNTKLib.Sigmoid(input);
                case Activation.Tanh:
                    return CNTKLib.Tanh(input);
            }
        }

        public static Function Conv2D(Variable input, int filterW, int filterH, int filterCount,
             int strideW = 1, int strideH = 1, uint seed = 34, string outputName = "")
        {
            if (input.Shape.Rank != 3)
            {
                throw new ArgumentException("Conv2D layer requires shape rank 3, got rank " + input.Shape.Rank);
            }

            // Assumes specific layout.
            var inputChannels = input.Shape[2];

            // Use GlorotUniform for weight initialization.
            var intializer = CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, seed);

            var convParams = new Parameter(new int[] { filterW, filterH, inputChannels, filterCount },
                    DataType, intializer, Device);
            var conv = CNTKLib.Convolution(convParams, input, new int[] { strideW, strideH, inputChannels });

            // Bias is initialized to 0.0.
            var bias = new Parameter(conv.Output.Shape, DataType, 0.0, Device);
            return CNTKLib.Plus(bias, conv);
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

        public static Function Reshape(Variable layer, int[] targetShape)
        {
            return CNTKLib.Reshape(layer, targetShape);
        }

        public static Function GlobalAveragePool2D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { layer.Shape[0], layer.Shape[1] });
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
