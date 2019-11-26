using System;

namespace SharpLearning.Neural.Test.Backends
{
    public enum DataType : uint
    {
        Float = 1,
        Double = 2,
        Int32 = 3,
        UInt8 = 4,
        Int16 = 5,
        Int8 = 6,
        String = 7,
        Complex64 = 8,
        Complex = 8,
        Int64 = 9,
        Bool = 10,
        QInt8 = 11,
        QUInt8 = 12,
        QInt32 = 13,
        BFloat16 = 14,
        QInt16 = 15,
        QUInt16 = 16,
        UInt16 = 17,
        Complex128 = 18,
        Half = 19,
        Resource = 20
    }

    public static class DataTypeExtensions
    {
        public static Type ToType(this DataType? type)
        {
            return type?.ToType();
        }

        public static Type ToType(this DataType type)
        {
            switch (type)
            {
                case DataType.Float:
                    return typeof(float);
                case DataType.Double:
                    return typeof(double);
            }

            throw new ArgumentOutOfRangeException(nameof(type));
        }
    }
}
