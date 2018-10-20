using System;
using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function MaxPool2D(this Function input,
            ValueTuple<int, int> poolShape,
            ValueTuple<int, int> strideShape,
            bool padding = false) 
        {
            var poolSizes = new int[] { poolShape.Item1, poolShape.Item2 };
            var strideSizes = new int[] { strideShape.Item1, strideShape.Item2 };
            var paddingSizes = new bool[] { padding, padding };

            return CNTKLib.Pooling(input,
                PoolingType.Max,
                poolSizes,
                strideSizes,
                paddingSizes);
        }
    }
}
