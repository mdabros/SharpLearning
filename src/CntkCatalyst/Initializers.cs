using System;
using CNTK;

namespace CntkCatalyst
{
    /// <summary>
    /// Initializer factory for CNTK
    /// </summary>
    public static class Initializers
    {
        public static CNTKDictionary Zero()
        {
            return CNTKLib.ConstantInitializer(0);
        }

        public static CNTKDictionary One()
        {
            return CNTKLib.ConstantInitializer(1);
        }

        public static CNTKDictionary None()
        {
            return null;
        }

        public static CNTKDictionary Uniform(uint seed = 1)
        {
            return CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale, seed);
        }

        public static CNTKDictionary Normal(uint seed = 1)
        {
            return CNTKLib.NormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }

        public static CNTKDictionary TruncatedNormal(uint seed = 1)
        {
            return CNTKLib.TruncatedNormalInitializer(CNTKLib.DefaultParamInitScale,
                seed);
        }

        public static CNTKDictionary Xavier(uint seed = 1)
        {
            return CNTKLib.XavierInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }

        public static CNTKDictionary GlorotNormal(uint seed = 1)
        {
            return CNTKLib.GlorotNormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }

        public static CNTKDictionary GlorotUniform(uint seed = 1)
        {
            return CNTKLib.GlorotUniformInitializer(CNTKLib.DefaultParamInitScale, 
                CNTKLib.SentinelValueForInferParamInitRank, 
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }

        public static CNTKDictionary HeNormal(uint seed = 1)
        {
            return CNTKLib.HeNormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }

        public static CNTKDictionary HeUniform(uint seed = 1)
        {
            return CNTKLib.HeUniformInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                seed);
        }
    }
}
