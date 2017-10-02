using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Serialization;
using System.IO;
using System.Collections.Generic;

namespace SharpLearning.InputOutput.Test.Serialization
{
    [TestClass]
    public class GenericBinarySerializerTest
    {
        [TestMethod]
        public void GenericBinarySerializer_Serialize()
        {
            var writer = new StringWriter();

            var sut = new GenericBinarySerializer();
            sut.Serialize(SerializationData, () => writer);

            var actual = writer.ToString();
            Assert.AreEqual(SerializationString, actual);
        }

        [TestMethod]
        public void GenericBinarySerializer_Deserialize()
        {
            var reader = new StringReader(SerializationString);

            var sut = new GenericBinarySerializer();
            var actual = sut.Deserialize<Dictionary<string, int>>(() => reader);

            CollectionAssert.AreEqual(SerializationData, actual);
        }

        Dictionary<string, int> SerializationData = new Dictionary<string, int>
        {
            {"Test1", 0}, {"Test2", 1}, {"Test3", 2}, {"Test4", 3}, {"Test5", 4}
        };

        readonly string SerializationString =
			"AAEAAAD/////AQAAAAAAAAAEAQAAAOEBU3lzdGVtLkNvbGxlY3Rpb25zLkdlbmVyaWMuRGljdGlvbmFyeWAyW1tTeXN0ZW0uU3RyaW5nLCBtc2NvcmxpYiwgVmVyc2lvbj00LjAuMC4wLCBDdWx0dXJlPW5ldXRyYWwsIFB1YmxpY0tleVRva2VuPWI3N2E1YzU2MTkzNGUwODldLFtTeXN0ZW0uSW50MzIsIG1zY29ybGliLCBWZXJzaW9uPTQuMC4wLjAsIEN1bHR1cmU9bmV1dHJhbCwgUHVibGljS2V5VG9rZW49Yjc3YTVjNTYxOTM0ZTA4OV1dBAAAAAdWZXJzaW9uCENvbXBhcmVyCEhhc2hTaXplDUtleVZhbHVlUGFpcnMAAwADCD5TeXN0ZW0uQ29sbGVjdGlvbnMuR2VuZXJpYy5Ob25SYW5kb21pemVkU3RyaW5nRXF1YWxpdHlDb21wYXJlcgjlAVN5c3RlbS5Db2xsZWN0aW9ucy5HZW5lcmljLktleVZhbHVlUGFpcmAyW1tTeXN0ZW0uU3RyaW5nLCBtc2NvcmxpYiwgVmVyc2lvbj00LjAuMC4wLCBDdWx0dXJlPW5ldXRyYWwsIFB1YmxpY0tleVRva2VuPWI3N2E1YzU2MTkzNGUwODldLFtTeXN0ZW0uSW50MzIsIG1zY29ybGliLCBWZXJzaW9uPTQuMC4wLjAsIEN1bHR1cmU9bmV1dHJhbCwgUHVibGljS2V5VG9rZW49Yjc3YTVjNTYxOTM0ZTA4OV1dW10FAAAACQIAAAAHAAAACQMAAAAEAgAAAD5TeXN0ZW0uQ29sbGVjdGlvbnMuR2VuZXJpYy5Ob25SYW5kb21pemVkU3RyaW5nRXF1YWxpdHlDb21wYXJlcgAAAAAHAwAAAAABAAAABQAAAAPjAVN5c3RlbS5Db2xsZWN0aW9ucy5HZW5lcmljLktleVZhbHVlUGFpcmAyW1tTeXN0ZW0uU3RyaW5nLCBtc2NvcmxpYiwgVmVyc2lvbj00LjAuMC4wLCBDdWx0dXJlPW5ldXRyYWwsIFB1YmxpY0tleVRva2VuPWI3N2E1YzU2MTkzNGUwODldLFtTeXN0ZW0uSW50MzIsIG1zY29ybGliLCBWZXJzaW9uPTQuMC4wLjAsIEN1bHR1cmU9bmV1dHJhbCwgUHVibGljS2V5VG9rZW49Yjc3YTVjNTYxOTM0ZTA4OV1dBPz////jAVN5c3RlbS5Db2xsZWN0aW9ucy5HZW5lcmljLktleVZhbHVlUGFpcmAyW1tTeXN0ZW0uU3RyaW5nLCBtc2NvcmxpYiwgVmVyc2lvbj00LjAuMC4wLCBDdWx0dXJlPW5ldXRyYWwsIFB1YmxpY0tleVRva2VuPWI3N2E1YzU2MTkzNGUwODldLFtTeXN0ZW0uSW50MzIsIG1zY29ybGliLCBWZXJzaW9uPTQuMC4wLjAsIEN1bHR1cmU9bmV1dHJhbCwgUHVibGljS2V5VG9rZW49Yjc3YTVjNTYxOTM0ZTA4OV1dAgAAAANrZXkFdmFsdWUBAAgGBQAAAAVUZXN0MQAAAAAB+v////z///8GBwAAAAVUZXN0MgEAAAAB+P////z///8GCQAAAAVUZXN0MwIAAAAB9v////z///8GCwAAAAVUZXN0NAMAAAAB9P////z///8GDQAAAAVUZXN0NQQAAAAL";
    }
}
