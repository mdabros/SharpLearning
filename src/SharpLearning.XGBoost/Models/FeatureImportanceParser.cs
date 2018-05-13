using System;

namespace SharpLearning.XGBoost.Models
{
    public static class FeatureImportanceParser
    {
        public static double[] ParseFromTreeDump(string[] textTrees, int numberOfFeatures)
        {
            var importanceType = new string[] { "gain=" };
            var newLine = new string[] { Environment.NewLine };

            var rawFeatureImportance = new double[numberOfFeatures];
            foreach (var tree in textTrees)
            {
                var lines = tree.Split(newLine, StringSplitOptions.RemoveEmptyEntries);
                foreach (var line in lines)
                {
                    if(!line.Contains("[") || line.Contains("booster"))
                    {
                        // Continue if line does not contain a split.
                        continue;
                    }

                    // Extract feature name from string between []
                    var name = line.Split('[')[1].Split(']')[0].Split('<')[0];

                    // Extract featureIndex
                    var featureIndex = int.Parse(name.Split('f')[1]);

                    // extract gain or cover
                    var gain = double.Parse(line.Split(importanceType, 
                        StringSplitOptions.RemoveEmptyEntries)[1].Split(',')[0]);

                    // add to featureImportance
                    rawFeatureImportance[featureIndex] += gain;
                }
            }

            return rawFeatureImportance;
        }               
    }
}
