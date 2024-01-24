using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Containers;
using SharpLearning.GradientBoost.GBMDecisionTree;

namespace SharpLearning.XGBoost.Models
{
    /// <summary>
    /// Conversion between xgboost trees text format and SharpLearning.GradientBoost GBMTrees.
    /// </summary>
    public static class XGBoostTreeConverter
    {
        static readonly string[] m_leafSplit = new string[] { "leaf=" };
        static readonly string[] m_yesSplit = new string[] { "yes=" };
        static readonly string[] m_noSplit = new string[] { "no=" };

        /// <summary>
        /// Parse array of feature importance values from text dump of XGBoost trees.
        /// </summary>
        /// <param name="textTrees"></param>
        /// <returns></returns>
        public static GBMTree[] FromXGBoostTextTreesToGBMTrees(string[] textTrees)
        {
            var trees = new GBMTree[textTrees.Length];
            for (int i = 0; i < textTrees.Length; i++)
            {
                trees[i] = ConvertXGBoostTextTreeToGBMTree(textTrees[i]);
            }

            return trees;
        }

        /// <summary>
        /// Converts a single XGBoost tree in text format to a GBMTree.
        /// </summary>
        /// <param name="textTree"></param>
        /// <returns></returns>
        public static GBMTree ConvertXGBoostTextTreeToGBMTree(string textTree)
        {
            List<GBMNode> nodes = ConvertXGBoostNodesToGBMNodes(textTree);

            return new GBMTree(nodes);
        }

        static List<GBMNode> ConvertXGBoostNodesToGBMNodes(string textTree)
        {
            var newLine = new string[] { "\n" };
            var lines = textTree.Split(newLine, StringSplitOptions.RemoveEmptyEntries);

            var nodes = new List<GBMNode>
            {
                // Add special root node for sharplearning
                new GBMNode
                {
                    FeatureIndex = -1,
                    SplitValue = -1,
                    LeftConstant = 0.5,
                    RightConstant = 0.5,
                },
            };

            // Order lines by node index and remove booster line.
            var ordered = lines.Where(l => !l.Contains("booster")).ToArray();

            var orderedLines = ordered
                .OrderBy(l => ParseNodeIndex(l))
                .ToDictionary(l => ParseNodeIndex(l), l => l);

            var nodeIndex = 1;
            foreach (var line in orderedLines.Values)
            {
                if (IsLeaf(line))
                {
                    // Leafs are not added as nodes, leaf values are included in the split nodes.
                    continue;
                }
                else
                {
                    var featureIndex = ParseFeatureIndex(line);
                    var splitValue = ParseSplitValue(line);
                    var yesIndex = ParseYesIndex(line);
                    var noIndex = ParseNoIndex(line);

                    var node = new GBMNode
                    {
                        FeatureIndex = featureIndex,
                        SplitValue = splitValue,
                        LeftConstant = -1,
                        LeftIndex = -1,
                        RightConstant = -1,
                        RightIndex = -1
                    };

                    var left = orderedLines[yesIndex];
                    if (IsLeaf(left))
                    {
                        node.LeftIndex = -1;
                        node.LeftConstant = ParseLeafValue(left);
                    }
                    else
                    {
                        nodeIndex++;
                        node.LeftIndex = nodeIndex;
                    }

                    var right = orderedLines[noIndex];
                    if (IsLeaf(right))
                    {
                        node.RightIndex = -1;
                        node.RightConstant = ParseLeafValue(right);
                    }
                    else
                    {
                        nodeIndex++;
                        node.RightIndex = nodeIndex;
                    }

                    nodes.Add(node);
                }
            }

            return nodes;
        }

        /// <summary>
        /// Checks if the current line contains a leaf node.
        /// </summary>
        /// <param name="line"></param>
        /// <returns></returns>
        public static bool IsLeaf(string line)
        {
            return line.Contains("leaf");
        }

        /// <summary>
        /// Parses the leaf value from a line.
        /// </summary>
        /// <param name="line"></param>
        /// <returns></returns>
        public static double ParseLeafValue(string line)
        {
            var valueLine = line.Split(m_leafSplit, StringSplitOptions.RemoveEmptyEntries)[1];
            var valueString = valueLine.Split(',')[0];
            var value = FloatingPointConversion.ToF64(valueString);
            return value;
        }

        /// <summary>
        /// Parses the feature index from a line.
        /// </summary>
        /// <param name="line"></param>
        /// <returns></returns>
        public static int ParseFeatureIndex(string line)
        {
            // Extract feature name from string between []
            var name = line.Split('[')[1].Split(']')[0].Split('<')[0];

            // Extract featureIndex
            var featureIndex = int.Parse(name.Split('f')[1]);

            return featureIndex;
        }

        /// <summary>
        /// Parses the split value from a line.
        /// </summary>
        /// <param name="line"></param>
        /// <returns></returns>
        public static double ParseSplitValue(string line)
        {
            // Extract feature value from string between []
            var valueString = line.Split('[')[1].Split(']')[0].Split('<')[1];

            // convert value.
            var value = FloatingPointConversion.ToF64(valueString);

            return value;
        }

        /// <summary>
        /// Parses the Yes node index from a line.
        /// </summary>
        /// <param name="line"></param>
        /// <returns></returns>
        public static int ParseYesIndex(string line)
        {
            return SplitYesNoIndex(line, m_yesSplit);
        }

        /// <summary>
        /// Parses the No index from a line.
        /// </summary>
        /// <param name="line"></param>
        /// <returns></returns>
        public static int ParseNoIndex(string line)
        {
            return SplitYesNoIndex(line, m_noSplit);
        }

        /// <summary>
        /// Parses the node index from a line.
        /// </summary>
        /// <param name="line"></param>
        /// <returns></returns>
        public static int ParseNodeIndex(string line)
        {
            var valueString = line.Split(':')[0];
            return int.Parse(valueString);
        }

        static int SplitYesNoIndex(string line, string[] separator)
        {
            var valueLine = line.Split(separator, StringSplitOptions.RemoveEmptyEntries)[1];
            var valueString = valueLine.Split(',')[0];
            var value = int.Parse(valueString);
            return value;
        }
    }
}
