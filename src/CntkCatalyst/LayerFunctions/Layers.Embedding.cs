using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        /// <summary>
        /// based on the Embedding from: https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/layers.py      
        /// </summary>
        public static Function Embedding(this Function input, int shape, CNTKDictionary initializer,
            DataType dataType, DeviceDescriptor device)
        {
            var weightsShape = new int[] { shape, CNTK.NDShape.InferredDimension };
            var weights = new Parameter( weightsShape, dataType, initializer, device);
            var result = CNTKLib.Times(weights, input);
            
            return result;
        }
    }
}
