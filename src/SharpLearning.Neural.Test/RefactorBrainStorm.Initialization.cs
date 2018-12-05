using System;
using MathNet.Numerics.Distributions;

namespace SharpLearning.Neural.Test.RefactorBranStorm
{
    /// <summary>
    /// 
    /// </summary>
    public struct FanInFanOut
    {
        /// <summary>
        /// The fan-in of the layer
        /// </summary>
        public readonly int FanIn;

        /// <summary>
        /// THe fan-out of the layer 
        /// </summary>
        public readonly int FanOut;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="fanIn"></param>
        /// <param name="fanOut"></param>
        public FanInFanOut(int fanIn, int fanOut)
        {
            FanIn = fanIn;
            FanOut = fanOut;
        }
    }

    /// <summary>
    /// Specifies the different types of initialization.
    /// </summary>
    public enum Initialization
    {
        /// <summary>
        /// Glorot initialization using uniform distribution, based on paper:
        /// http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        /// </summary>
        GlorotUniform,

        /// <summary>
        /// He initialization using uniform distribution, based on paper:
        /// https://arxiv.org/pdf/1502.01852.pdf
        /// </summary>
        HeUniform,

        /// <summary>
        /// Glorot initialization using normal distribution, based on paper:
        /// http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        /// </summary>
        GlorotNormal,

        /// <summary>
        /// He initialization using normal distribution, based on paper:
        /// https://arxiv.org/pdf/1502.01852.pdf
        /// </summary>
        HeNormal,
    }

    /// <summary>
    /// Weight initialization.
    /// </summary>
    public static class WeightInitialization
    {
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
                    return (float)Math.Sqrt(6.0 / (double)(fans.FanIn + fans.FanOut));
                case Initialization.HeUniform:
                    return (float)Math.Sqrt(2.0 / (double)fans.FanIn);
                case Initialization.GlorotNormal:
                    return (float)Math.Sqrt(6.0 / (double)(fans.FanIn + fans.FanOut));
                case Initialization.HeNormal:
                    return (float)Math.Sqrt(2.0 / (double)fans.FanIn);
                default:
                    throw new ArgumentException("Unsupported Initialization type: " + initialization);
            }
        }

        /// <summary>
        /// Calculates the distribution
        /// </summary>
        /// <param name="initialization"></param>
        /// <param name="fans"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static IContinuousDistribution GetWeightDistribution(Initialization initialization, FanInFanOut fans, Random random)
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
    }
}
