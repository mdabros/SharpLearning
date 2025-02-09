using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.GradientBoost.GBMDecisionTree;
using SharpLearning.XGBoost.Models;

namespace SharpLearning.XGBoost.Test.Models;

[TestClass]
public class XGBoostTreeConverterTest
{
    readonly double m_delta = 0.000001;

    [TestMethod]
    public void XGBoostTreeConverter_ConvertXGBoostTextTreeToGBMTree_Tree_1()
    {
        var actualNodes = XGBoostTreeConverter.ConvertXGBoostTextTreeToGBMTree(Tree1).Nodes;
        var expectedNodes = Tree1Nodes;

        Assert.AreEqual(expectedNodes.Count, actualNodes.Count);

        for (var i = 0; i < expectedNodes.Count; i++)
        {
            var expected = expectedNodes[i];
            var actual = actualNodes[i];
            AssertGBMNode(expected, actual);
        }
    }

    [TestMethod]
    public void XGBoostTreeConverter_ConvertXGBoostTextTreeToGBMTree_Tree_2()
    {
        var actualNodes = XGBoostTreeConverter.ConvertXGBoostTextTreeToGBMTree(Tree2).Nodes;
        var expectedNodes = Tree2Nodes;

        Assert.AreEqual(expectedNodes.Count, actualNodes.Count);

        for (var i = 0; i < expectedNodes.Count; i++)
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

    const string Tree1 = @"booster[0]
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

    static readonly List<GBMNode> Tree1Nodes =
    [
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
            RightConstant = 0.3,
            LeftIndex = 6,
            RightIndex = -1
        },

        /*3*/ new GBMNode
        {
            FeatureIndex = 8,
            SplitValue = 0.13,
            LeftConstant = 0.404167,
            RightConstant = 0.1,
            LeftIndex = -1,
            RightIndex = -1
        },

        /*4*/ new GBMNode
        {
            FeatureIndex = 4,
            SplitValue = 72.205,
            LeftConstant = 0.0444444,
            RightConstant = 0.205882,
            LeftIndex = -1,
            RightIndex = -1
        },

        /*5*/ new GBMNode
        {
            FeatureIndex = 0,
            SplitValue = 1.51715,
            LeftConstant = 0.0527778,
            RightConstant = -0.0020202,
            LeftIndex = -1,
            RightIndex = -1
        },
    ];

    const string Tree2 = @"booster[10]
0:[f2<2.545] yes=1,no=2,missing=1,gain=46.9086,cover=214
	1:[f1<13.785] yes=3,no=4,missing=3,gain=12.7152,cover=60
		3:[f1<13.495] yes=7,no=8,missing=7,gain=4.75871,cover=24
			7:leaf=0.0747358,cover=19
			8:leaf=-0.0296133,cover=5
		4:[f8<0.135] yes=9,no=10,missing=9,gain=3.60802,cover=36
			9:leaf=0.154654,cover=35
			10:leaf=-0.0200209,cover=1
	2:[f6<5.83] yes=5,no=6,missing=5,gain=5.34865,cover=154
		5:leaf=0.140797,cover=2
		6:[f0<1.51715] yes=11,no=12,missing=11,gain=3.39987,cover=152
			11:leaf=0.0257847,cover=54
			12:leaf=-0.00524031,cover=98";

    static readonly List<GBMNode> Tree2Nodes =
    [
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
            SplitValue = 2.545,
            LeftConstant = -1,
            RightConstant = -1,
            LeftIndex = 2,
            RightIndex = 3
        },

       /*1*/ new GBMNode
        {
            FeatureIndex = 1,
            SplitValue = 13.785,
            LeftConstant = -1,
            RightConstant = -1,
            LeftIndex = 4,
            RightIndex = 5
        },

       /*2*/ new GBMNode
        {
            FeatureIndex = 6,
            SplitValue = 5.83,
            LeftConstant = 0.140797,
            RightConstant = -1,
            LeftIndex = -1,
            RightIndex = 6
        },

        /*3*/ new GBMNode
        {
            FeatureIndex = 1,
            SplitValue = 13.495,
            LeftConstant = 0.0747358,
            RightConstant = -0.0296133,
            LeftIndex = -1,
            RightIndex = -1
        },

        /*4*/ new GBMNode
        {
            FeatureIndex = 8,
            SplitValue = 0.135,
            LeftConstant = 0.154654,
            RightConstant = -0.0200209,
            LeftIndex = -1,
            RightIndex = -1
        },

        /*5*/ new GBMNode
        {
            FeatureIndex = 0,
            SplitValue = 1.51715,
            LeftConstant = 0.0257847,
            RightConstant = -0.00524031,
            LeftIndex = -1,
            RightIndex = -1
        },
    ];
}
