using System;
using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function Conv2D(this Function input, int filterW, int filterH, int filterCount,
             DataType d, DeviceDescriptor device,
             int strideW = 1, int strideH = 1, uint seed = 34, string outputName = "")
        {
            if (input.Output.Shape.Rank != 3)
            {
                throw new ArgumentException("Conv2D layer requires shape rank 3, got rank " + input.Output.Shape.Rank);
            }

            var inputChannels = input.Output.Shape[2];

            var convWScale = 0.26;
            var convParams = new Parameter(new int[] { filterW, filterH, inputChannels, filterCount }, d,
                CNTKLib.GlorotUniformInitializer(convWScale, -1, 2, seed), device);

            return CNTKLib.Convolution(convParams, input, new int[] { strideW, strideH, inputChannels });
        }
    }
}
