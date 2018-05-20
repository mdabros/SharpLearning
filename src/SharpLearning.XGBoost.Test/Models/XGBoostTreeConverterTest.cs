using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.GradientBoost.GBMDecisionTree;
using SharpLearning.XGBoost.Models;

namespace SharpLearning.XGBoost.Test.Models
{
    [TestClass]
    public class XGBoostTreeConverterTest
    {
        readonly double m_delta = 0.0000001;

        [TestMethod]
        public void XGBoostTreeConverter_ConvertXGBoostTextTreeToGBMTree()
        {
            var actualNodes = XGBoostTreeConverter.ConvertXGBoostTextTreeToGBMTree(m_tree1).Nodes;
            var expectedNodes = m_tree1Nodes;

            Assert.AreEqual(expectedNodes.Count, actualNodes.Count);

            for (int i = 0; i < expectedNodes.Count; i++)
            {
                var expected = expectedNodes[i];
                var actual = actualNodes[i];
                AssertGBMNode(expected, actual);
            }
        }

        [TestMethod]
        public void XGBoostTreeConverter_IsLeaf()
        {
            Assert.IsTrue(XGBoostTreeConverter.IsLeaf("7:leaf=0.404167,cover=35"));
            Assert.IsFalse(XGBoostTreeConverter.IsLeaf("0:[f2<2.695] yes=1,no=2,missing=1,gain=343.922,cover=214"));
        }

        [TestMethod]
        public void XGBoostTreeConverter_ParseLeafValue()
        {
            var actual = XGBoostTreeConverter.ParseLeafValue("7:leaf=0.404167,cover=35");
            var expected = 0.404167;

            Assert.AreEqual(expected, actual, m_delta);
        }

        [TestMethod]
        public void XGBoostTreeConverter_ParseFeatureIndex()
        {
            var actual = XGBoostTreeConverter.ParseFeatureIndex("0:[f2<2.695] yes=1,no=2,missing=1,gain=343.922,cover=214");
            var expected = 2;

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void XGBoostTreeConverter_ParseSplitValue()
        {
            var actual = XGBoostTreeConverter.ParseSplitValue("0:[f2<2.695] yes=1,no=2,missing=1,gain=343.922,cover=214");
            var expected = 2.695;

            Assert.AreEqual(expected, actual, m_delta);
        }

        [TestMethod]
        public void XGBoostTreeConverter_ParseYesIndex()
        {
            var actual = XGBoostTreeConverter.ParseYesIndex("0:[f2<2.695] yes=1,no=2,missing=1,gain=343.922,cover=214");
            var expected = 1;

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void XGBoostTreeConverter_ParseNoIndex()
        {
            var actual = XGBoostTreeConverter.ParseNoIndex("0:[f2<2.695] yes=1,no=2,missing=1,gain=343.922,cover=214");
            var expected = 2;

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void XGBoostTreeConverter_ParseNodeIndex()
        {
            var actual = XGBoostTreeConverter.ParseNodeIndex("1:[f2<2.695] yes=1,no=2,missing=1,gain=343.922,cover=214");
            var expected = 1;

            Assert.AreEqual(expected, actual);
        }

        void AssertGBMNode(GBMNode expected, GBMNode actual)
        {
            Assert.AreEqual(expected.Depth, actual.Depth);
            Assert.AreEqual(expected.FeatureIndex, actual.FeatureIndex);
            Assert.AreEqual(expected.LeftConstant, actual.LeftConstant, m_delta);
            Assert.AreEqual(expected.LeftError, actual.LeftError, m_delta);
            Assert.AreEqual(expected.LeftIndex, actual.LeftIndex);
            Assert.AreEqual(expected.RightConstant, actual.RightConstant, m_delta);
            Assert.AreEqual(expected.RightError, actual.RightError, m_delta);
            Assert.AreEqual(expected.RightIndex, actual.RightIndex);
            Assert.AreEqual(expected.SampleCount, actual.SampleCount);
            Assert.AreEqual(expected.SplitValue, actual.SplitValue, m_delta);
        }

        readonly string[] m_textTrees = new string[] { m_tree1, m_tree2 };

        static readonly string m_tree1 = @"booster[0]
0:[f2<2.695] yes=1,no=2,missing=1,gain=343.922,cover=214
	1:[f6<9.81] yes=3,no=4,missing=3,gain=74.1261,cover=61
		3:[f8<0.13] yes=7,no=8,missing=7,gain=10.7401,cover=37
			7:leaf=0.404167,cover=35
			8:leaf=0.1,cover=2
		4:[f4<72.205] yes=9,no=10,missing=9,gain=12.9966,cover=24
			9:leaf=0.0444444,cover=8
			10:leaf=0.205882,cover=16
	2:[f5<1.28] yes=5,no=6,missing=5,gain=23.4366,cover=153
		5:[f0<1.51715] yes=11,no=12,missing=11,gain=10.462,cover=151
			11:leaf=0.0527778,cover=53
			12:leaf=-0.0020202,cover=98
		6:leaf=0.3,cover=2";


        static readonly string m_tree2 = @"booster[1]
0:[f2<2.695] yes=1,no=2,missing=1,gain=280.77,cover=214
	1:[f6<9.81] yes=3,no=4,missing=3,gain=60.1843,cover=61
		3:[f8<0.13] yes=7,no=8,missing=7,gain=8.27457,cover=37
			7:leaf=0.364873,cover=35
			8:leaf=0.0933333,cover=2
		4:[f0<1.52174] yes=9,no=10,missing=9,gain=10.9722,cover=24
			9:leaf=0.19627,cover=14
			10:leaf=0.0537255,cover=10
	2:[f5<1.28] yes=5,no=6,missing=5,gain=20.5033,cover=153
		5:[f0<1.51715] yes=11,no=12,missing=11,gain=8.50854,cover=151
			11:leaf=0.0475977,cover=53
			12:leaf=-0.00182022,cover=98
		6:leaf=0.28,cover=2";

        static readonly List<GBMNode> m_tree1Nodes = new List<GBMNode>
        {
           /*-1*/ new GBMNode
            {
                FeatureIndex = -1,
                SplitValue = -1,
                LeftConstant = 0.5,
                RightConstant = 0.5,
            },

           /*0*/ new GBMNode
            {
                FeatureIndex = 2,
                SplitValue = 2.695,
                LeftConstant = -1,
                RightConstant = -1,
                LeftIndex = 2,
                RightIndex = 3
            },

           /*1*/ new GBMNode
            {
                FeatureIndex = 6,
                SplitValue = 9.81,
                LeftConstant = -1,
                RightConstant = -1,
                LeftIndex = 4,
                RightIndex = 5
            },

           /*2*/ new GBMNode
            {
                FeatureIndex = 5,
                SplitValue = 1.28,
                LeftConstant = -1,
                RightConstant = -1,
                LeftIndex = 6,
                RightIndex = 7
            },

            /*3*/ new GBMNode
            {
                FeatureIndex = 8,
                SplitValue = 0.13,
                LeftConstant = -1,
                RightConstant = -1,
                LeftIndex = 8,
                RightIndex = 9
            },

            /*4*/ new GBMNode
            {
                FeatureIndex = 4,
                SplitValue = 72.205,
                LeftConstant = -1,
                RightConstant = -1,
                LeftIndex = 10,
                RightIndex = 11
            },

            /*5*/ new GBMNode
            {
                FeatureIndex = 0,
                SplitValue = 1.51715,
                LeftConstant = -1,
                RightConstant = -1,
                LeftIndex = 12,
                RightIndex = 13
            },

            /*6*/ new GBMNode
            {
                FeatureIndex = -1,
                SplitValue = -1,
                LeftConstant = 0.3,
                RightConstant = 0.3,
            },

            /*7*/ new GBMNode
            {
                FeatureIndex = -1,
                SplitValue = -1,
                LeftConstant = 0.404167,
                RightConstant = 0.404167,
            },

            /*8*/ new GBMNode
            {
                FeatureIndex = -1,
                SplitValue = -1,
                LeftConstant = 0.1,
                RightConstant = 0.1,
            },

            /*9*/ new GBMNode
            {
                FeatureIndex = -1,
                SplitValue = -1,
                LeftConstant = 0.0444444,
                RightConstant = 0.0444444,
            },

            /*10*/ new GBMNode
            {
                FeatureIndex = -1,
                SplitValue = -1,
                LeftConstant = 0.205882,
                RightConstant = 0.205882,
            },

            /*11*/ new GBMNode
            {
                FeatureIndex = -1,
                SplitValue = -1,
                LeftConstant = 0.0527778,
                RightConstant = 0.0527778,
            },

            /*12*/ new GBMNode
            {
                FeatureIndex = -1,
                SplitValue = -1,
                LeftConstant = -0.0020202,
                RightConstant = -0.0020202,
            },
        };  
    }
}
