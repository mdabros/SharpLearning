using System;
using CNTK;

namespace SharpLearning.Cntk.Test
{
    /// <summary>
    /// Initializes for cntk
    /// </summary>
    public enum Initializer
    {
        Uniform,
        Normal,
        TruncatedNormal,
        Zeros,
        Ones,
        Constant,
        Xavier,
        GlorotNormal,
        GlorotUniform,
        HeNormal,
        HeUniform,
    }

    /// <summary>
    /// Initializer factory for CNTK
    /// </summary>
    public static class Initializers
    {
        public static CNTKDictionary Create(Initializer initializer)
        {
            switch (initializer)
            {
                case Initializer.Uniform:
                    return Uniform();
                case Initializer.Normal:
                    return Normal();
                case Initializer.TruncatedNormal:
                    return TruncatedNormal();
                case Initializer.Zeros:
                    return CNTKLib.ConstantInitializer(0);
                case Initializer.Ones:
                    return CNTKLib.ConstantInitializer(1);
                case Initializer.Xavier:
                    return Xavier();
                case Initializer.GlorotNormal:
                    return GlorotNormal();
                case Initializer.GlorotUniform:
                    return GlorotUniform();
                case Initializer.HeNormal:
                    return HeNormal();
                case Initializer.HeUniform:
                    return HeUniform();
                default:
                    throw new ArgumentException("Unsupported initializer: " + initializer);
            }
        }

        static CNTKDictionary Uniform(uint seed = 1)
        {
            return CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale, seed);
        }

        static CNTKDictionary Normal(uint seed = 1)
        {
            return CNTKLib.NormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }

        static CNTKDictionary TruncatedNormal(uint seed = 1)
        {
            return CNTKLib.TruncatedNormalInitializer(CNTKLib.DefaultParamInitScale,
                seed);
        }

        static CNTKDictionary Xavier(uint seed = 1)
        {
            return CNTKLib.XavierInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }

        static CNTKDictionary GlorotNormal(uint seed = 1)
        {
            return CNTKLib.GlorotNormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }

        static CNTKDictionary GlorotUniform(uint seed = 1)
        {
            return CNTKLib.GlorotUniformInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }

        static CNTKDictionary HeNormal(uint seed = 1)
        {
            return CNTKLib.HeNormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }

        static CNTKDictionary HeUniform(uint seed = 1)
        {
            return CNTKLib.HeUniformInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }
    }
}