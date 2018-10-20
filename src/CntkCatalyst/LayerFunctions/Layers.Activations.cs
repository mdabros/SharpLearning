using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function ReLU(this Function input)
        {
            return CNTKLib.ReLU(input);
        }

        public static Function Sigmoid(this Function input)
        {
            return CNTKLib.Sigmoid(input);
        }

        public static Function Softmax(this Function input)
        {
            return CNTKLib.Softmax(input);
        }
    }
}
