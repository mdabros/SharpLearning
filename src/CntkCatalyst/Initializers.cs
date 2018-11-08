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

        public static CNTKDictionary Uniform(int seed)
        {
            return CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale, (uint)seed);
        }

        public static CNTKDictionary Normal(int seed)
        {
            return CNTKLib.NormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary TruncatedNormal(int seed)
        {
            return CNTKLib.TruncatedNormalInitializer(CNTKLib.DefaultParamInitScale,
                (uint)seed);
        }

        public static CNTKDictionary Xavier(int seed)
        {
            return CNTKLib.XavierInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary GlorotNormal(int seed)
        {
            return CNTKLib.GlorotNormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary GlorotUniform(int seed)
        {
            return CNTKLib.GlorotUniformInitializer(CNTKLib.DefaultParamInitScale, 
                CNTKLib.SentinelValueForInferParamInitRank, 
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary HeNormal(int seed)
        {
            return CNTKLib.HeNormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary HeUniform(int seed)
        {
            return CNTKLib.HeUniformInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }
    }
}
