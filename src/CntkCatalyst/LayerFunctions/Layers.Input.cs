using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function Input(NDShape inputShape, DataType dataType, bool isSparse = false, string name = "")
        {
            return Variable.InputVariable(inputShape, dataType, isSparse: isSparse);
        }
    }
}
