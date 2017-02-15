using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.InputOutput.Test.Csv
{
    [TestClass]
    public class DictionaryExtensionsTest
    {
        [TestMethod]
        public void DictionaryExtensions_GetValues()
        {
            var sut = new Dictionary<string, int> { { "F1", 0 }, { "F2", 1 } };
            var expected = new int[] {0, 1};
            var actual = sut.GetValues(new string[] {"F1", "F2" });
            Assert.AreNotEqual(expected, actual);
        }
    }
}
