using System;
using CNTK;

namespace SharpLearning.Backend.Cntk
{
    public static class CntkExtensions
    {
        public static CNTK.DataType ToCntk(this DataType dataType)
        {
            switch (dataType)
            {
                //case DataType.Half:
                    //return CNTK.DataType;
                case DataType.Single:
                    return CNTK.DataType.Float;
                case DataType.Double:
                    return CNTK.DataType.Double;
                case DataType.UInt8:
                    return CNTK.DataType.UChar;
                //case DataType.Int8:
                //    return CNTK.DataType.Int8;
                //case DataType.UInt16:
                //    return CNTK.DataType.UInt16;
                //case DataType.Int16:
                //    return CNTK.DataType.Int16;
                //case DataType.UInt32: // Not supported?
                    //return CNTK.DataType.UI;
                //case DataType.Int32:
                //    return CNTK.DataType.Int32;
                //case DataType.UInt64: // Not supported?
                    //return CNTK.DataType.UI;
                //case DataType.Int64:
                //    return CNTK.DataType.Int64;
                default:
                    throw new NotSupportedException(dataType.ToString());
            }
        }

        // Allocates twice
        public static NDShape ToCntkShape(this ReadOnlySpan<int> shape)
        {
            return NDShape.CreateNDShape(shape.ToArray());
        }
    }
}
