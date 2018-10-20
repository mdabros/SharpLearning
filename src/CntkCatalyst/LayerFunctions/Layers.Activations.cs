using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function ReLU(this Function x)
        {
            return CNTKLib.ReLU(x);
        }

        public static Function Sigmoid(this Function x)
        {
            return CNTKLib.Sigmoid(x);
        }

        public static Function Softmax(this Function x)
        {
            return CNTKLib.Softmax(x);
        }
    }
}
