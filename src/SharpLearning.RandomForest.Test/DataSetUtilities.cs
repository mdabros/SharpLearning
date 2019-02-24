using System.IO;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;
using SharpLearning.RandomForest.Test.Properties;

namespace SharpLearning.RandomForest.Test
{
    public static class DataSetUtilities
    {
        public static (F64Matrix observations, double[] targets) LoadAptitudeDataSet()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            return (observations, targets);
        }
    }
}
