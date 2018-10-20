using System;
using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function Conv2D(this Function input,
            ValueTuple<int, int> filterSize,
            int filterCount,
            ValueTuple<int, int> strideSize,
            DeviceDescriptor device,
            DataType dataType,
            uint seed = 34,
            string outputName = "")
        {
            if (input.Output.Shape.Rank != 3)
            {
                throw new ArgumentException("Conv2D layer requires shape rank 3, got rank " + input.Output.Shape.Rank);
            }

            var inputChannels = input.Output.Shape[2];

            var convWScale = 0.26;
            var convParams = new Parameter(new int[] { filterSize.Item1, filterSize.Item2, inputChannels, filterCount }, dataType,
                CNTKLib.GlorotUniformInitializer(convWScale, -1, 2, seed), device);

            return CNTKLib.Convolution(convParams, input, new int[] { strideSize.Item1, strideSize.Item2, inputChannels });
        }
    }
}
