using System;
using CNTK;

namespace SharpLearning.Neural.Cntk
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
        public static CNTKDictionary Create(Initializer initializer,
            double scale,
            uint seed = 42)
        {
            switch (initializer)
            {
                case Initializer.Uniform:
                    return CNTKLib.UniformInitializer(scale, seed);
                case Initializer.Normal:
                case Initializer.TruncatedNormal:
                case Initializer.Zeros:
                case Initializer.Ones:
                case Initializer.Constant:
                case Initializer.Xavier:
                case Initializer.GlorotNormal:
                case Initializer.GlorotUniform:
                case Initializer.HeNormal:
                case Initializer.HeUniform:
                default:
                    throw new ArgumentException("Unsupported initializer: " + initializer);
            }
        }
    }
}
