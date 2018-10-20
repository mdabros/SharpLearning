using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function Input(NDShape inputShape, DataType d)
        {
            return Variable.InputVariable(inputShape, d); ;
        }
    }
}
