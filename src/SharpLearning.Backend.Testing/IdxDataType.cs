using System;

namespace SharpLearning.Backend.Testing
{
    public enum IdxDataType : byte
    {
        Byte   = 0x08, //: unsigned byte 
        SByte  = 0x09, //: signed byte 
        Short  = 0x0B, //: short (2 bytes) 
        Int    = 0x0C, //: int (4 bytes) 
        Float  = 0x0D, //: float (4 bytes) 
        Double = 0x0E, //: double (8 bytes)
    }

    public static class IdxDataTypeExtensions
    {
        public static Type ToType(this IdxDataType type)
        {
            switch (type)
            {
                case IdxDataType.Byte:
                    return typeof(byte);
                case IdxDataType.SByte:
                    return typeof(sbyte);
                case IdxDataType.Short:
                    return typeof(short);
                case IdxDataType.Int:
                    return typeof(int);
                case IdxDataType.Float:
                    return typeof(float);
                case IdxDataType.Double:
                    return typeof(double);
                default:
                    throw new NotSupportedException(type.ToString());
            }
        }
    }
}
