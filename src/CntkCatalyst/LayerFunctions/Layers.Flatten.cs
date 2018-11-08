using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function Flatten(this Function input)
        {
            return CNTKLib.Flatten(input);
        }
    }
}
