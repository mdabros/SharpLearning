using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function GlobalMaxPooling1D(this Function input)
        {
            // _Pooling(PoolingType_Max, NDShape.unknown().dimensions(), pad=False, op_name='GlobalMaxPooling', name=name)

            //def _Pooling(op,           # PoolingType_Max or _Average
            // filter_shape, # shape of receptive field, e.g. (3,3)
            // sequential= False, # pooling in time if True (filter_shape[0] corresponds to dynamic axis)
            // strides= 1,
            // pad= False,
            // op_name= None,
            // name= ''):

            // return _Pooling(PoolingType_Max, NDShape.unknown().dimensions(), pad=False, op_name='GlobalMaxPooling', name=name)

            return CNTKLib.Pooling(input, 
                PoolingType.Max, 
                NDShape.Unknown(), 
                new int[] { 1 },
                new bool[] { false });
        }
    }
}
