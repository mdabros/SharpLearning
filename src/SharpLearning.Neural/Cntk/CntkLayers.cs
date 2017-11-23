using System;
using System.Linq;
using CNTK;

namespace SharpLearning.Neural.Cntk
{
    public enum Activation
    {
        None,
        ReLU,
        LeakyReLU,
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

        public static Function Activation(Variable input, Activation activation)
        {
            switch (activation)
            {
                default:
                case Cntk.Activation.None:
                    return input;
                case Cntk.Activation.ReLU:
                    return CNTKLib.ReLU(input);
                case Cntk.Activation.LeakyReLU:
                    return CNTKLib.LeakyReLU(input);
                case Cntk.Activation.Sigmoid:
                    return CNTKLib.Sigmoid(input);
                case Cntk.Activation.Tanh:
                    return CNTKLib.Tanh(input);
            }
        }

        public static Function Conv2D(Variable input, int filterW, int filterH, int filterCount,
             int strideW = 1, int strideH = 1, uint seed = 34, string outputName = "")
        {
            if (input.Shape.Rank != 3)
            {
                throw new ArgumentException($"Conv2D layer requires shape rank 3, got rank {input.Shape.Rank}");
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
                throw new ArgumentException($"Pool2D layer requires shape rank 3, got rank {input.Shape.Rank}");
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

        /// <summary>
        /// Resnet blocks from
        /// https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py
        /// </summary>
        public static class ResNet
        {
            public static Function IdentityBlock(Variable input, int filterW, int filterH,
                int filterCount1, int filterCount2, int filterCount3)
            {
                var layer = CntkLayers.Conv2D(input, 1, 1, filterCount1);
                layer = CntkLayers.BatchNormalizationLayer(layer, true);
                layer = CntkLayers.Activation(layer, Cntk.Activation.ReLU);

                layer = CntkLayers.Conv2D(layer, filterW, filterH, filterCount2);
                layer = CntkLayers.BatchNormalizationLayer(layer, true);
                layer = CntkLayers.Activation(layer, Cntk.Activation.ReLU);

                layer = CntkLayers.Conv2D(layer, 1, 1, filterCount3);
                layer = CntkLayers.BatchNormalizationLayer(layer, true);

                layer = CNTKLib.Plus(layer, input);
                return CntkLayers.Activation(layer, Cntk.Activation.ReLU);
            }

            public static Function ConvolutionBlock(Variable input, int filterW, int filterH,
                int filterCount1, int filterCount2, int filterCount3,
                int strideW = 2, int strideH = 2)
            {
                var layer = CntkLayers.Conv2D(input, 1, 1, filterCount1, strideW, strideH);
                layer = CntkLayers.BatchNormalizationLayer(layer, true);
                layer = CntkLayers.Activation(layer, Cntk.Activation.ReLU);

                layer = CntkLayers.Conv2D(layer, filterW, filterH, filterCount2);
                layer = CntkLayers.BatchNormalizationLayer(layer, true);
                layer = CntkLayers.Activation(layer, Cntk.Activation.ReLU);

                layer = CntkLayers.Conv2D(layer, 1, 1, filterCount3);
                layer = CntkLayers.BatchNormalizationLayer(layer, true);

                var shortcut = CntkLayers.Conv2D(input, 1, 1, filterCount3, strideW, strideH);
                shortcut = CntkLayers.BatchNormalizationLayer(shortcut, true);

                layer = CNTKLib.Plus(layer, shortcut);
                return CNTKLib.ReLU(layer);
            }
        }
    }
}
