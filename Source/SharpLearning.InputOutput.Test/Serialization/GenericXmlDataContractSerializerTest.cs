using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Serialization;
using System.Collections.Generic;
using System.IO;

namespace SharpLearning.InputOutput.Test.Serialization
{
    [TestClass]
    public class GenericXmlDataContractSerializerTest
    {
        [TestMethod]
        public void GenericXmlDataContractSerializer_Serialize()
        {
            var writer = new StringWriter();
            
            GenericXmlDataContractSerializer.Serialize(SerializationData,
                () => writer);

            Assert.AreEqual(SerializationString, writer.ToString());
        }

        [TestMethod]
        public void GenericXmlDataContractSerializer_Deserialize()
        {
            var reader = new StringReader(SerializationString);

            var actual = GenericXmlDataContractSerializer.Deserialize<Dictionary<string, int>>(
                () => reader);

            CollectionAssert.AreEqual(SerializationData, actual);
        }

        Dictionary<string, int> SerializationData = new Dictionary<string, int>
        {
            {"Test1", 0}, {"Test2", 1}, {"Test3", 2}, {"Test4", 3}, {"Test5", 4}
        };

        readonly string SerializationString =
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<ArrayOfKeyValueOfstringint xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" z:Size=\"5\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\">\r\n  <KeyValueOfstringint>\r\n    <Key z:Id=\"2\">Test1</Key>\r\n    <Value>0</Value>\r\n  </KeyValueOfstringint>\r\n  <KeyValueOfstringint>\r\n    <Key z:Id=\"3\">Test2</Key>\r\n    <Value>1</Value>\r\n  </KeyValueOfstringint>\r\n  <KeyValueOfstringint>\r\n    <Key z:Id=\"4\">Test3</Key>\r\n    <Value>2</Value>\r\n  </KeyValueOfstringint>\r\n  <KeyValueOfstringint>\r\n    <Key z:Id=\"5\">Test4</Key>\r\n    <Value>3</Value>\r\n  </KeyValueOfstringint>\r\n  <KeyValueOfstringint>\r\n    <Key z:Id=\"6\">Test5</Key>\r\n    <Value>4</Value>\r\n  </KeyValueOfstringint>\r\n</ArrayOfKeyValueOfstringint>";
    }
}
