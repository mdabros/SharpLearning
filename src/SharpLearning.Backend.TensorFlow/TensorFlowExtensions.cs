using System;
using TensorFlow;

namespace SharpLearning.Backend.TensorFlow
{
    public static class TensorFlowExtensions
    {
        public static TFDataType ToTF(this DataType dataType)
        {
            switch (dataType)
            {
                case DataType.Half:
                    return TFDataType.Half;
                case DataType.Single:
                    return TFDataType.Float;
                case DataType.Double:
                    return TFDataType.Double;
                case DataType.UInt8:
                    return TFDataType.UInt8;
                case DataType.Int8:
                    return TFDataType.Int8;
                case DataType.UInt16:
                    return TFDataType.UInt16;
                case DataType.Int16:
                    return TFDataType.Int16;
                //case DataType.UInt32: // Not supported?
                    //return TFDataType.UI;
                case DataType.Int32:
                    return TFDataType.Int32;
                //case DataType.UInt64: // Not supported?
                    //return TFDataType.UI;
                case DataType.Int64:
                    return TFDataType.Int64;
                default:
                    throw new NotSupportedException(dataType.ToString());
            }
        }

        // Allocates twice
        public static TFShape ToTFShape(this ReadOnlySpan<int> shape)
        {
            long[] longShape = new long[shape.Length];
            return new TFShape(longShape);
        }
    }
}
