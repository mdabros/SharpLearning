using System;
using MathNet.Numerics.Distributions;
using SharpLearning.Neural.Layers;

namespace SharpLearning.Neural.Initializations
{
    /// <summary>
    /// Weight initialization.
    /// </summary>
    public static class WeightInitialization
    {
        /// <summary>
        /// Calculates the fan-in and fan-out used for weight initialization.
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="inputDepth"></param>
        /// <returns></returns>
        public static FanInFanOut GetFans(ILayer layer, int inputWidth, int inputHeight, int inputDepth)
        {
            if (layer is Conv2DLayer conv)
            {
                var receptiveFieldSize = conv.FilterWidth * conv.FilterHeight;

                var fanIn = inputDepth * receptiveFieldSize;
                var fanOut = layer.Depth * receptiveFieldSize;

                return new FanInFanOut(fanIn, fanOut);

            }
            else if (layer is DenseLayer)
            {
                var fanIn = inputWidth * inputHeight * inputDepth;
                var fanOut = layer.Width * layer.Height * layer.Depth;
                return new FanInFanOut(fanIn, fanOut);
            }
            else // default case
            {
                var fanIn = (int)Math.Sqrt(inputWidth * inputHeight * inputDepth);
                var fanOut = (int)Math.Sqrt(layer.Width * layer.Height * layer.Depth);

                return new FanInFanOut(fanIn, fanOut);

            }
        }

        /// <summary>
        /// Calculates the distribution
        /// </summary>
        /// <param name="initialization"></param>
        /// <param name="fans"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static IContinuousDistribution GetWeightDistribution(Initialization initialization, 
            FanInFanOut fans, Random random)
        {
            var bound = InitializationBound(initialization, fans);

            switch (initialization)
            {
                case Initialization.GlorotUniform:
                    return new ContinuousUniform(-bound, bound, new Random(random.Next()));
                case Initialization.HeUniform:
                    return new ContinuousUniform(-bound, bound, new Random(random.Next()));
                case Initialization.GlorotNormal:
                    return new Normal(0.0, bound, new Random(random.Next()));
                case Initialization.HeNormal:
                    return new Normal(0.0, bound, new Random(random.Next()));
                default:
                    throw new ArgumentException("Unsupported Initialization type: " + initialization);
            }
        }

        /// <summary>
        /// Returns the default initialization bounds based on the initialization type, fan-in and fan-out.
        /// </summary>
        /// <param name="initialization"></param>
        /// <param name="fans"></param>
        /// <returns></returns>
        public static float InitializationBound(Initialization initialization, FanInFanOut fans)
        {
            switch (initialization)
            {
                case Initialization.GlorotUniform:
                    return (float)Math.Sqrt(6.0 / (fans.FanIn + fans.FanOut));
                case Initialization.HeUniform:
                    return (float)Math.Sqrt(2.0 / fans.FanIn);
                case Initialization.GlorotNormal:
                    return (float)Math.Sqrt(6.0 / (fans.FanIn + fans.FanOut));
                case Initialization.HeNormal:
                    return (float)Math.Sqrt(2.0 / fans.FanIn);
                default:
                    throw new ArgumentException("Unsupported Initialization type: " + initialization);
            }
        }
    }
}
