using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function Conv1D(this Function input,
            int filterShape,
            int filterCount,
            int strideShape,
            CNTKDictionary weightInitializer,
            CNTKDictionary biasInitializer,
            DeviceDescriptor device,
            DataType dataType)
        {

            int[] filterSizes;
            BoolVector autoPadding;

            var inputShape = input.Output.Shape;
            if (inputShape.Rank > 1)
            {
                var inputChannels = inputShape[inputShape.Rank - 1];
                filterSizes = new int[]
                {
                    filterShape,
                    inputChannels,
                    filterCount
                };

                autoPadding = CreateFilledBoolVector(filterSizes.Length, false);
            }
            else
            {
                filterSizes = new int[]
                {
                    filterShape,
                    filterCount
                };

                autoPadding = CreateFilledBoolVector(1, false);
            }

            var filterStrides = new int[] { strideShape };
            var sharing = CreateFilledBoolVector(filterStrides.Length, true);
            var dilation = new int[] { 1 };

            var weights = new Parameter(NDShape.CreateNDShape(filterSizes), dataType,
                   weightInitializer, device);

            var result = CNTKLib.Convolution(weights, input, filterStrides,
                sharing, autoPadding, dilation);

            if (biasInitializer != null)
            {
                // Bias dimensions should be defined for filter dimensions.
                // For instance for 2D case: (1, 1, filterChannels).
                var biasShape = new List<int> { 1 }; //filterStrides.Select(s => 1).ToList();
                biasShape.Add(filterCount);

                var bias = new Parameter(NDShape.CreateNDShape(biasShape.ToArray()),
                    dataType, biasInitializer, device);

                result = CNTKLib.Plus(result, bias);
            }

            return result;
        }

        internal static BoolVector CreateFilledBoolVector(int size, bool fill)
        {
            var boolVector = new BoolVector(size);
            for (int i = 0; i < size; i++)
            {
                boolVector.Add(fill);
            }
            return boolVector;
        }
    }
}
