using System.Collections.Generic;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Serialization;

namespace SharpLearning.InputOutput.Test.Serialization
{
    [TestClass]
    public class GenericBinarySerializerTest
    {
        readonly Dictionary<string, int> m_serializationData = new Dictionary<string, int>
        {
            {"Test1", 0}, {"Test2", 1}, {"Test3", 2}, {"Test4", 3}, {"Test5", 4}
        };

        readonly string m_serializationString = "AAEAAAD/////AQAAAAAAAAAEAQAAAOEBU3lzdGVtLkNvbGxlY3Rpb25zLkdlbmVyaWMuRGljdGlvbmFyeWAyW1tTeXN0ZW0uU3RyaW5nLCBtc2NvcmxpYiwgVmVyc2lvbj00LjAuMC4wLCBDdWx0dXJlPW5ldXRyYWwsIFB1YmxpY0tleVRva2VuPWI3N2E1YzU2MTkzNGUwODldLFtTeXN0ZW0uSW50MzIsIG1zY29ybGliLCBWZXJzaW9uPTQuMC4wLjAsIEN1bHR1cmU9bmV1dHJhbCwgUHVibGljS2V5VG9rZW49Yjc3YTVjNTYxOTM0ZTA4OV1dBAAAAAdWZXJzaW9uCENvbXBhcmVyCEhhc2hTaXplDUtleVZhbHVlUGFpcnMAAwADCJIBU3lzdGVtLkNvbGxlY3Rpb25zLkdlbmVyaWMuR2VuZXJpY0VxdWFsaXR5Q29tcGFyZXJgMVtbU3lzdGVtLlN0cmluZywgbXNjb3JsaWIsIFZlcnNpb249NC4wLjAuMCwgQ3VsdHVyZT1uZXV0cmFsLCBQdWJsaWNLZXlUb2tlbj1iNzdhNWM1NjE5MzRlMDg5XV0I5QFTeXN0ZW0uQ29sbGVjdGlvbnMuR2VuZXJpYy5LZXlWYWx1ZVBhaXJgMltbU3lzdGVtLlN0cmluZywgbXNjb3JsaWIsIFZlcnNpb249NC4wLjAuMCwgQ3VsdHVyZT1uZXV0cmFsLCBQdWJsaWNLZXlUb2tlbj1iNzdhNWM1NjE5MzRlMDg5XSxbU3lzdGVtLkludDMyLCBtc2NvcmxpYiwgVmVyc2lvbj00LjAuMC4wLCBDdWx0dXJlPW5ldXRyYWwsIFB1YmxpY0tleVRva2VuPWI3N2E1YzU2MTkzNGUwODldXVtdBQAAAAkCAAAABwAAAAkDAAAABAIAAACSAVN5c3RlbS5Db2xsZWN0aW9ucy5HZW5lcmljLkdlbmVyaWNFcXVhbGl0eUNvbXBhcmVyYDFbW1N5c3RlbS5TdHJpbmcsIG1zY29ybGliLCBWZXJzaW9uPTQuMC4wLjAsIEN1bHR1cmU9bmV1dHJhbCwgUHVibGljS2V5VG9rZW49Yjc3YTVjNTYxOTM0ZTA4OV1dAAAAAAcDAAAAAAEAAAAFAAAAA+MBU3lzdGVtLkNvbGxlY3Rpb25zLkdlbmVyaWMuS2V5VmFsdWVQYWlyYDJbW1N5c3RlbS5TdHJpbmcsIG1zY29ybGliLCBWZXJzaW9uPTQuMC4wLjAsIEN1bHR1cmU9bmV1dHJhbCwgUHVibGljS2V5VG9rZW49Yjc3YTVjNTYxOTM0ZTA4OV0sW1N5c3RlbS5JbnQzMiwgbXNjb3JsaWIsIFZlcnNpb249NC4wLjAuMCwgQ3VsdHVyZT1uZXV0cmFsLCBQdWJsaWNLZXlUb2tlbj1iNzdhNWM1NjE5MzRlMDg5XV0E/P///+MBU3lzdGVtLkNvbGxlY3Rpb25zLkdlbmVyaWMuS2V5VmFsdWVQYWlyYDJbW1N5c3RlbS5TdHJpbmcsIG1zY29ybGliLCBWZXJzaW9uPTQuMC4wLjAsIEN1bHR1cmU9bmV1dHJhbCwgUHVibGljS2V5VG9rZW49Yjc3YTVjNTYxOTM0ZTA4OV0sW1N5c3RlbS5JbnQzMiwgbXNjb3JsaWIsIFZlcnNpb249NC4wLjAuMCwgQ3VsdHVyZT1uZXV0cmFsLCBQdWJsaWNLZXlUb2tlbj1iNzdhNWM1NjE5MzRlMDg5XV0CAAAAA2tleQV2YWx1ZQEACAYFAAAABVRlc3QxAAAAAAH6/////P///wYHAAAABVRlc3QyAQAAAAH4/////P///wYJAAAABVRlc3QzAgAAAAH2/////P///wYLAAAABVRlc3Q0AwAAAAH0/////P///wYNAAAABVRlc3Q1BAAAAAs=";

        [TestMethod]
        public void GenericBinarySerializer_Serialize()
        {
            var writer = new StringWriter();

            var sut = new GenericBinarySerializer();
            sut.Serialize(m_serializationData, () => writer);

            var actual = writer.ToString();
            Assert.AreEqual(m_serializationString, actual);
        }

        [TestMethod]
        public void GenericBinarySerializer_Deserialize()
        {
            var reader = new StringReader(m_serializationString);

            var sut = new GenericBinarySerializer();
            var actual = sut.Deserialize<Dictionary<string, int>>(() => reader);

            CollectionAssert.AreEqual(m_serializationData, actual);
        }
    }
}
